import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from vhmodels.utils import lima_utils
from vhmodels.vh_checker import backends, cli, factory
from vhmodels.vh_checker.backends import ApptainerBackend
from vhmodels.vh_checker.factory import load_model
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    RESULT_MARKER,
)


@pytest.fixture
def isolated_apptainer_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "uv-cache"
    monkeypatch.setenv("VHMODELS_APPTAINER_CACHE_DIR", str(cache_path))
    return cache_path.resolve()


def _frame(obj):
    return f"{RESULT_MARKER}{json.dumps(obj)}{RESULT_MARKER}\n"


def test_apptainer_build_command_uses_lima_on_macos(monkeypatch, tmp_path):
    workdir = tmp_path / "working directory"
    workdir.mkdir()
    image_path = tmp_path / "images with spaces" / "prottrans.sif"
    monkeypatch.chdir(workdir)

    backend = ApptainerBackend(image_path, use_lima=True)
    command = backend.build_command(["--project", "prottrans"])

    assert command[:7] == [
        "limactl",
        "shell",
        "--tty=false",
        "--preserve-env",
        "--workdir",
        str(workdir.resolve()),
        lima_utils.LIMA_INSTANCE,
    ]
    assert command[7:] == [
        "apptainer",
        "exec",
        str(image_path),
        "/opt/venv/bin/python",
        "-m",
        "vhmodels.vh_checker.embed",
        "--project",
        "prottrans",
    ]


def test_apptainer_runtime_lookup_depends_on_launcher(monkeypatch):
    looked_up = []

    def fake_which(executable):
        looked_up.append(executable)
        return f"/bin/{executable}"

    monkeypatch.setattr(backends.shutil, "which", fake_which)

    assert ApptainerBackend("model.sif", use_lima=False).is_runtime_available()
    assert ApptainerBackend("model.sif", use_lima=True).is_runtime_available()
    assert looked_up == ["apptainer", "limactl"]


def test_apptainer_lima_env_binds_host_home(monkeypatch, tmp_path):
    monkeypatch.setattr(backends.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPTAINER_BINDPATH", "/data")
    monkeypatch.setenv("SINGULARITYENV_PYTHONPATH", "/host/source")

    env = ApptainerBackend("model.sif", use_lima=True).subprocess_env()

    assert env["APPTAINER_BINDPATH"] == f"/data,{tmp_path.resolve()}"
    assert "SINGULARITYENV_PYTHONPATH" not in env


def test_apptainer_lima_prepare_requires_shared_workdir(monkeypatch):
    shared = iter([True, False])
    monkeypatch.setattr(
        lima_utils, "is_lima_shared_path", lambda path: next(shared)
    )
    monkeypatch.setattr(
        lima_utils,
        "ensure_lima_instance",
        lambda: pytest.fail(
            "VM should not start for an inaccessible working directory"
        ),
    )

    backend = ApptainerBackend("model.sif", use_lima=True)

    with pytest.raises(RuntimeError, match="relative input paths"):
        backend.prepare()


def test_ensure_lima_creates_fast_rosetta_vm(monkeypatch):
    commands = []
    monkeypatch.setattr(
        lima_utils.shutil, "which", lambda executable: "/bin/limactl"
    )
    monkeypatch.setattr(lima_utils.platform, "machine", lambda: "arm64")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if command[:2] == ["limactl", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lima_utils.subprocess, "run", fake_run)

    lima_utils.ensure_lima_instance()

    assert commands[0][0] == ["limactl", "list", "--format=json"]
    assert commands[1][0] == [
        "limactl",
        "start",
        "--tty=false",
        f"--name={lima_utils.LIMA_INSTANCE}",
        "--vm-type=vz",
        "--arch=aarch64",
        "--rosetta",
        "--mount-writable",
        "template:apptainer",
    ]
    assert commands[1][1]["check"] is True


def test_ensure_lima_detects_python_running_under_rosetta(monkeypatch):
    commands = []
    monkeypatch.setattr(
        lima_utils.shutil, "which", lambda executable: "/bin/limactl"
    )
    monkeypatch.setattr(lima_utils.platform, "machine", lambda: "x86_64")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["limactl", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[0] == "sysctl":
            return subprocess.CompletedProcess(command, 0, stdout="1\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lima_utils.subprocess, "run", fake_run)

    lima_utils.ensure_lima_instance()

    start_command = next(command for command in commands if command[1] == "start")
    assert "--arch=aarch64" in start_command
    assert "--rosetta" in start_command


@pytest.mark.parametrize(
    ("status", "expected_starts"),
    [("Running", 0), ("Stopped", 1)],
)
def test_ensure_lima_reuses_existing_vm(monkeypatch, status, expected_starts):
    commands = []
    monkeypatch.setattr(
        lima_utils.shutil, "which", lambda executable: "/bin/limactl"
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["limactl", "list"]:
            output = '{"name":"vhmodels-apptainer","status":"' + status + '"}\n'
            return subprocess.CompletedProcess(command, 0, stdout=output)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lima_utils.subprocess, "run", fake_run)

    lima_utils.ensure_lima_instance()

    start_commands = [command for command in commands if command[1] == "start"]
    assert len(start_commands) == expected_starts
    if start_commands:
        assert start_commands[0] == [
            "limactl",
            "start",
            "--tty=false",
            lima_utils.LIMA_INSTANCE,
        ]


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
        lima_utils.LIMA_INSTANCE,
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


def test_modelproxy_apptainer_prepares_and_uses_lima(monkeypatch, tmp_path):
    captured = {}
    image_path = tmp_path / "dinobloom.sif"
    image_path.touch()

    def fake_run_subprocess(cmd, payload, subprocess_env, timeout):
        captured["cmd"] = cmd
        captured["payload"] = payload
        return (_frame({"output": [42]}), "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lima_utils, "uses_lima", lambda: True)
    monkeypatch.setattr(factory, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        backends.ApptainerBackend,
        "is_runtime_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        backends.ApptainerBackend,
        "prepare",
        lambda self: captured.setdefault("prepared", True),
    )

    model = load_model(
        "dinobloom", runtime="apptainer", image_path=image_path, device="cpu"
    )
    result = model.embed("image.bmp")

    assert captured["prepared"] is True
    assert captured["cmd"][:7] == [
        "limactl",
        "shell",
        "--tty=false",
        "--preserve-env",
        "--workdir",
        str(tmp_path.resolve()),
        lima_utils.LIMA_INSTANCE,
    ]
    assert captured["cmd"][7:10] == [
        "apptainer",
        "exec",
        str(image_path.resolve()),
    ]
    assert [json.loads(line) for line in captured["payload"].splitlines()] == [
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "load_kwargs": {"device": "cpu"},
        },
        {MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": "image.bmp"},
    ]
    assert result == [42]
