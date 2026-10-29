import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from vhmodels.utils import lima_utils
from vhmodels.vh_checker import cli, factory


@pytest.fixture(autouse=True)
def isolated_uv_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "uv-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(cache_path))
    return cache_path.resolve()


def _missing_apptainer(*args, **kwargs):
    raise FileNotFoundError


def _failed_build(command, **kwargs):
    raise subprocess.CalledProcessError(1, command)


def _missing_lima():
    raise RuntimeError("Lima is not installed; install it with 'brew install lima'.")


@pytest.mark.parametrize(
    ("project", "host", "existing_output", "patches", "message"),
    [
        ("dinobloom", "linux-x86_64", False,
         {"subprocess.run": _missing_apptainer}, "Apptainer is not installed"),
        ("dinobloom", "windows-amd64", False, {}, "Linux or on macOS with Lima"),
        ("dinobloom", "linux-x86_64", True,
         {"_check_apptainer_installed": lambda: True}, "already exists"),
        ("dinobloom", "linux-x86_64", False,
         {"_check_apptainer_installed": lambda: True,
          "subprocess.run": _failed_build}, "Failed to build Apptainer image"),
        ("dinobloom", "macos-arm64", False,
         {"ensure_lima_instance": _missing_lima}, "brew install lima"),
        # The output directory is shared with Lima, the uv build cache is not.
        ("dinobloom", "macos-arm64", False,
         {"is_lima_shared_path": lambda path, shared=iter([True, False]): next(shared)},
         "uv build cache must be under your home directory"),
    ],
)  # fmt: skip
def test_create_apptainer_image_reports_errors(
    monkeypatch, tmp_path, project, host, existing_output, patches, message
):
    output_path = tmp_path / f"{project}.sif"
    if existing_output:
        output_path.touch()
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: host)
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: True)
    # Unless a case expects it, neither a build nor a Lima VM may start.
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("build started")
    )
    monkeypatch.setattr(
        lima_utils, "ensure_lima_instance", lambda: pytest.fail("Lima started")
    )
    for name, replacement in patches.items():
        if name == "subprocess.run":
            monkeypatch.setattr(cli.subprocess, "run", replacement)
        elif hasattr(lima_utils, name):
            monkeypatch.setattr(lima_utils, name, replacement)
        else:
            monkeypatch.setattr(cli, name, replacement)

    result = CliRunner().invoke(
        cli.main,
        ["create-apptainer-image", project, "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert message in result.output


def test_create_apptainer_image_renders_and_builds_definition(
    monkeypatch, tmp_path, isolated_uv_cache
):
    captured = {}
    output_path = tmp_path / "dinobloom.sif"
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(cli, "_check_apptainer_installed", lambda: True)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["definition"] = Path(command[-1]).read_text(encoding="utf-8")
        captured["staged_registry"] = (
            kwargs["cwd"] / "vhmodels" / "models" / "registry.py"
        ).is_file()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.main,
        [
            "create-apptainer-image",
            "dinobloom",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["command"][:-1] == [
        "apptainer",
        "build",
        "--bind",
        f"{isolated_uv_cache}:/opt/uv-cache",
        str(output_path.resolve()),
    ]
    assert isolated_uv_cache.is_dir()
    assert captured["kwargs"]["check"] is True
    assert captured["staged_registry"] is True
    # The image installs the model's uv requirements and serves the worker.
    assert "models/DinoBloom/requirements.linux-x86_64.txt" in captured["definition"]
    assert "python -m vhmodels.vh_checker.worker serve" in captured["definition"]
    assert "export UV_CACHE_DIR=/opt/uv-cache" in captured["definition"]
    for placeholder in (
        "{project}",
        "{model_dir}",
        "{python_version}",
        "{requirements_file}",
        "{torch_backend_arg}",
        "{exclude_arg}",
    ):
        assert placeholder not in captured["definition"]
    assert f"Using uv build cache '{isolated_uv_cache}'" in result.output
    assert "Successfully created Apptainer image" in result.output


@pytest.mark.parametrize(
    ("project", "model_dir", "torch_backend", "exclude_file"),
    [
        ("hyformer", "Hyformer", "cu118", "requirements-exclude.linux-x86_64.txt"),
        ("prottrans", "ProtTrans", None, None),
    ],
)
def test_create_apptainer_image_uses_model_uv_dependencies(
    monkeypatch,
    tmp_path,
    project,
    model_dir,
    torch_backend,
    exclude_file,
):
    captured = {}
    output_path = tmp_path / f"{project}.sif"
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(cli, "_check_apptainer_installed", lambda: True)

    def fake_run(command, **kwargs):
        captured["definition"] = Path(command[-1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.main,
        ["create-apptainer-image", project, "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert f"models/{model_dir}/requirements.linux-x86_64.txt" in captured["definition"]
    assert '"cpython-3.11.15-linux-x86_64-gnu"' in captured["definition"]
    assert 'requirements.linux-x86_64.txt "ray[serve]==2.59.0"' in captured["definition"]
    if torch_backend is None:
        assert "--torch-backend" not in captured["definition"]
    else:
        assert f"--torch-backend {torch_backend}" in captured["definition"]
    if exclude_file is None:
        assert "--excludes" not in captured["definition"]
    else:
        assert (
            f"--excludes /opt/vhmodels-src/vhmodels/models/{model_dir}/{exclude_file}"
            in captured["definition"]
        )


def test_create_apptainer_image_uses_lima_and_amd64_on_macos(
    monkeypatch, tmp_path, isolated_uv_cache
):
    captured = {}
    output_path = tmp_path / "hyformer.sif"
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "macos-arm64")
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: True)
    monkeypatch.setattr(
        lima_utils,
        "ensure_lima_instance",
        lambda: captured.setdefault("prepared", True),
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["definition"] = Path(command[-1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.main,
        ["create-apptainer-image", "hyformer", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert captured["prepared"] is True
    assert captured["command"][:7] == [
        "limactl",
        "shell",
        "--tty=false",
        "--preserve-env",
        "--workdir",
        str(captured["kwargs"]["cwd"].resolve()),
        "vhmodels-apptainer",
    ]
    assert captured["command"][7:13] == [
        "env",
        "APPTAINER_TMPDIR=/var/tmp",
        "apptainer",
        "build",
        "--arch",
        "amd64",
    ]
    assert captured["command"][13:15] == [
        "--bind",
        f"{isolated_uv_cache}:/opt/uv-cache",
    ]
    assert captured["command"][15] == str(output_path.resolve())
    assert captured["kwargs"]["cwd"].parent == output_path.parent
    assert "models/Hyformer/requirements.linux-x86_64.txt" in captured["definition"]
    assert '"cpython-3.11.15-linux-x86_64-gnu"' in captured["definition"]
    assert "--torch-backend cu118" in captured["definition"]


def test_run_cli_forwards_options_and_reports_model_errors(monkeypatch, tmp_path):
    captured = {}
    image_path = tmp_path / "custom.sif"

    class FakeModel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            captured["closed"] = True

        def embed(self, data):
            captured["data"] = data
            return [1, 2, 3]

    def fake_load_model(project, runtime, image_path):
        captured["project"] = project
        captured["runtime"] = runtime
        captured["image_path"] = image_path
        return FakeModel()

    monkeypatch.setattr(factory, "load_model", fake_load_model)

    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "dinobloom",
            '{"path": "image.bmp"}',
            "--runtime",
            "apptainer",
            "--image-path",
            str(image_path),
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "project": "dinobloom",
        "runtime": "apptainer",
        "image_path": image_path,
        "data": {"path": "image.bmp"},
        "closed": True,
    }
    assert "[\n  1,\n  2,\n  3\n]" in result.output

    class FailingModel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def embed(self, data):
            raise RuntimeError("The Apptainer image does not exist.")

    monkeypatch.setattr(factory, "load_model", lambda *args, **kwargs: FailingModel())

    result = CliRunner().invoke(
        cli.main,
        ["run", "nicheformer", "{}", "--runtime", "apptainer"],
    )

    assert result.exit_code == 1
    assert "Error: The Apptainer image does not exist." in result.output
