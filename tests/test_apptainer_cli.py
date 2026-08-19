import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from vhmodels.utils import lima_utils
from vhmodels.vh_checker import cli, factory


@pytest.fixture(autouse=True)
def isolated_apptainer_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "uv-cache"
    monkeypatch.setenv("VHMODELS_APPTAINER_CACHE_DIR", str(cache_path))
    return cache_path.resolve()


def test_apptainer_cache_uses_xdg_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VHMODELS_APPTAINER_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))

    assert (
        cli._apptainer_uv_cache_dir()
        == (tmp_path / "xdg-cache" / "vhmodels" / "apptainer" / "uv").resolve()
    )


def test_create_apptainer_image_reports_missing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "linux-x86_64")

    def missing_apptainer(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", missing_apptainer)

    result = CliRunner().invoke(
        cli.main,
        [
            "create-apptainer-image",
            "dinobloom",
            "--output",
            str(tmp_path / "dinobloom.sif"),
        ],
    )

    assert result.exit_code == 1
    assert "Apptainer is not installed" in result.output


def test_create_apptainer_image_rejects_unsupported_host(monkeypatch):
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "windows-amd64")
    called = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: called.append(args)
    )

    result = CliRunner().invoke(cli.main, ["create-apptainer-image", "dinobloom"])

    assert result.exit_code == 1
    assert "Linux or on macOS with Lima" in result.output
    assert called == []


def test_create_apptainer_image_rejects_unknown_project():
    result = CliRunner().invoke(cli.main, ["create-apptainer-image", "unknown"])

    assert result.exit_code == 1
    assert "is not registered" in result.output


def test_create_apptainer_image_renders_and_builds_definition(
    monkeypatch, tmp_path, isolated_apptainer_cache
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
        f"{isolated_apptainer_cache}:/opt/vhmodels-build-cache",
        str(output_path.resolve()),
    ]
    assert isolated_apptainer_cache.is_dir()
    assert captured["kwargs"]["check"] is True
    assert captured["staged_registry"] is True
    assert "From: ghcr.io/astral-sh/uv:0.12.3" in captured["definition"]
    assert "From: ubuntu:24.04" in captured["definition"]
    assert "/uv /usr/local/bin/uv" in captured["definition"]
    assert "models/DinoBloom/requirements.linux-x86_64.txt" in captured["definition"]
    assert "vhmodels /opt/vhmodels-src/vhmodels" in captured["definition"]
    assert '"cpython-3.10-linux-x86_64-gnu"' in captured["definition"]
    assert "--torch-backend cu126" in captured["definition"]
    assert "uv pip install --python /opt/venv/bin/python" in captured["definition"]
    assert (
        "PYTHONPATH=/opt/vhmodels-src /opt/venv/bin/python -c "
        '"import vhmodels.models.DinoBloom.model"' in captured["definition"]
    )
    assert "export VIRTUAL_ENV=/opt/venv" in captured["definition"]
    assert "export PATH=/opt/venv/bin:$PATH" in captured["definition"]
    assert "export PYTHONPATH=/opt/vhmodels-src" in captured["definition"]
    assert "%startscript" in captured["definition"]
    assert "python -m vhmodels.vh_checker.worker serve" in captured["definition"]
    assert "export TMPDIR=/var/tmp" in captured["definition"]
    assert (
        'mkdir -p "$APPTAINER_ROOTFS/opt/vhmodels-build-cache"'
        in captured["definition"]
    )
    assert "export UV_CACHE_DIR=/opt/vhmodels-build-cache" in captured["definition"]
    assert (
        "export UV_PYTHON_CACHE_DIR=/opt/vhmodels-build-cache/python"
        in captured["definition"]
    )
    assert "export UV_LINK_MODE=copy" in captured["definition"]
    assert "uv cache clean" not in captured["definition"]
    assert "micromamba" not in captured["definition"]
    for placeholder in (
        "{project}",
        "{model_dir}",
        "{python_version}",
        "{requirements_file}",
        "{torch_backend_arg}",
        "{exclude_arg}",
    ):
        assert placeholder not in captured["definition"]
    assert f"Using uv build cache '{isolated_apptainer_cache}'" in result.output
    assert "Successfully created Apptainer image" in result.output


@pytest.mark.parametrize(
    ("project", "model_dir", "python_version", "torch_backend", "exclude_file"),
    [
        ("dinobloom", "DinoBloom", "3.10", "cu126", None),
        (
            "hyformer",
            "Hyformer",
            "3.9",
            "cu118",
            "requirements-exclude.linux-x86_64.txt",
        ),
        ("mole", "MolE", "3.10", "cu126", None),
        ("prottrans", "ProtTrans", "3.11", None, None),
    ],
)
def test_create_apptainer_image_uses_model_uv_dependencies(
    monkeypatch,
    tmp_path,
    project,
    model_dir,
    python_version,
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
    assert f'"cpython-{python_version}-linux-x86_64-gnu"' in captured["definition"]
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


def test_create_apptainer_image_does_not_overwrite(monkeypatch, tmp_path):
    output_path = tmp_path / "existing.sif"
    output_path.touch()
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(cli, "_check_apptainer_installed", lambda: True)
    called = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: called.append(args)
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "create-apptainer-image",
            "dinobloom",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert called == []


def test_create_apptainer_image_reports_build_failure(monkeypatch, tmp_path):
    output_path = tmp_path / "dinobloom.sif"
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(cli, "_check_apptainer_installed", lambda: True)

    def failed_build(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli.subprocess, "run", failed_build)

    result = CliRunner().invoke(
        cli.main,
        [
            "create-apptainer-image",
            "dinobloom",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "Failed to build Apptainer image" in result.output


def test_create_apptainer_image_uses_lima_and_amd64_on_macos(
    monkeypatch, tmp_path, isolated_apptainer_cache
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
        f"{isolated_apptainer_cache}:/opt/vhmodels-build-cache",
    ]
    assert captured["command"][15] == str(output_path.resolve())
    assert captured["kwargs"]["cwd"].parent == output_path.parent
    assert "models/Hyformer/requirements.linux-x86_64.txt" in captured["definition"]
    assert '"cpython-3.9-linux-x86_64-gnu"' in captured["definition"]
    assert "--torch-backend cu118" in captured["definition"]


def test_create_apptainer_image_reports_missing_lima(monkeypatch, tmp_path):
    output_path = tmp_path / "dinobloom.sif"
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "macos-arm64")
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: True)

    def missing_lima():
        raise RuntimeError(
            "Lima is not installed; install it with 'brew install lima'."
        )

    monkeypatch.setattr(lima_utils, "ensure_lima_instance", missing_lima)

    result = CliRunner().invoke(
        cli.main,
        ["create-apptainer-image", "dinobloom", "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert "brew install lima" in result.output


def test_create_apptainer_image_rejects_unshared_macos_cache(monkeypatch, tmp_path):
    output_path = tmp_path / "dinobloom.sif"
    shared_paths = iter([True, False])
    monkeypatch.setattr(cli, "_determine_current_platform", lambda: "macos-arm64")
    monkeypatch.setattr(
        lima_utils, "is_lima_shared_path", lambda path: next(shared_paths)
    )
    monkeypatch.setattr(
        lima_utils,
        "ensure_lima_instance",
        lambda: pytest.fail("Lima should not start for an inaccessible cache"),
    )

    result = CliRunner().invoke(
        cli.main,
        ["create-apptainer-image", "dinobloom", "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert "build cache must be under your home directory" in result.output


def test_run_cli_forwards_apptainer_runtime_and_image(monkeypatch, tmp_path):
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
