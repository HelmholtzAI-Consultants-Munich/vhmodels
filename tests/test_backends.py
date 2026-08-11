# Unit tests for the runtime backend abstraction (vhmodels.vh_checker.backends).
import pytest

from vhmodels.vh_checker import backends
from vhmodels.vh_checker.backends import ApptainerBackend, CondaBackend, get_backend


# --- get_backend -----------------------------------------------------------


def test_get_backend_conda():
    backend = get_backend("conda", "vhmodels-dinobloom")
    assert isinstance(backend, CondaBackend)
    assert backend.env_name == "vhmodels-dinobloom"


def test_get_backend_apptainer():
    backend = get_backend("apptainer", "/images/vhmodels-dinobloom.sif")
    assert isinstance(backend, ApptainerBackend)
    assert backend.image_path == "/images/vhmodels-dinobloom.sif"


def test_get_backend_unknown_runtime():
    with pytest.raises(NotImplementedError):
        get_backend("docker", "vhmodels-dinobloom")


# --- CondaBackend ----------------------------------------------------------


def test_conda_build_command():
    backend = CondaBackend("vhmodels-prottrans")
    cmd = backend.build_command(["--project", "prottrans", "--model", "xl"])
    assert cmd == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "vhmodels-prottrans",
        "python",
        "-m",
        "vhmodels.vh_checker.embed",
        "--project",
        "prottrans",
        "--model",
        "xl",
    ]


def test_conda_build_command_forwards_stdin():
    # --no-capture-output must be present, else `conda run` drops stdin and the
    # runner reads an empty document.
    cmd = CondaBackend("vhmodels-mole").build_command(["--project", "mole"])
    assert "--no-capture-output" in cmd


def test_conda_build_command_does_not_mutate_input():
    backend = CondaBackend("vhmodels-mole")
    script_args = ["--project", "mole"]
    backend.build_command(script_args)
    assert script_args == ["--project", "mole"]  # list(...) copy, not in place


def test_conda_is_available_true(monkeypatch):
    class FakeResult:
        stdout = "# conda environments:\nbase\nvhmodels-mole\n"

    monkeypatch.setattr(backends.subprocess, "run", lambda *a, **k: FakeResult())
    assert CondaBackend("vhmodels-mole").is_available() is True


def test_conda_is_available_false(monkeypatch):
    class FakeResult:
        stdout = "# conda environments:\nbase\n"

    monkeypatch.setattr(backends.subprocess, "run", lambda *a, **k: FakeResult())
    assert CondaBackend("vhmodels-mole").is_available() is False


def test_conda_subprocess_env_strips_pythonpath(monkeypatch):
    # vhmodels is installed into the env, so PYTHONPATH is unnecessary and a
    # host entry could shadow the env's packages -- it must be removed.
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    env = CondaBackend("vhmodels-mole").subprocess_env()
    assert "PYTHONPATH" not in env


def test_conda_subprocess_env_passes_other_vars_through(monkeypatch):
    # Everything except PYTHONPATH is preserved for the model's benefit.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = CondaBackend("vhmodels-mole").subprocess_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


# --- ApptainerBackend ------------------------------------------------------


def test_apptainer_build_command():
    backend = ApptainerBackend(
        "/images/vhmodels-prottrans.sif",
        use_lima=False,
    )
    cmd = backend.build_command(["--project", "prottrans", "--model", "xl"])
    assert cmd == [
        "apptainer",
        "exec",
        "/images/vhmodels-prottrans.sif",
        "/opt/venv/bin/python",
        "-m",
        "vhmodels.vh_checker.embed",
        "--project",
        "prottrans",
        "--model",
        "xl",
    ]


def test_apptainer_is_available_for_sif_file(tmp_path):
    image_path = tmp_path / "vhmodels-mole.sif"
    image_path.touch()
    assert ApptainerBackend(image_path).is_available() is True


def test_apptainer_is_unavailable_for_missing_path_or_directory(tmp_path):
    missing_path = tmp_path / "missing.sif"
    directory_path = tmp_path / "directory.sif"
    directory_path.mkdir()
    assert ApptainerBackend(missing_path).is_available() is False
    assert ApptainerBackend(directory_path).is_available() is False


def test_apptainer_subprocess_env_strips_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    monkeypatch.setenv("APPTAINERENV_PYTHONPATH", "/override/apptainer")
    monkeypatch.setenv("APPTAINER_BINDPATH", "/data")
    env = ApptainerBackend("vhmodels-mole.sif", use_lima=False).subprocess_env()
    assert "PYTHONPATH" not in env
    assert "APPTAINERENV_PYTHONPATH" not in env
    assert env["APPTAINER_BINDPATH"] == "/data"


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
        backends.LIMA_INSTANCE,
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
    monkeypatch.setattr(backends, "is_lima_shared_path", lambda path: next(shared))
    monkeypatch.setattr(
        backends,
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
    monkeypatch.setattr(backends.shutil, "which", lambda executable: "/bin/limactl")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if command[:2] == ["limactl", "list"]:
            return backends.subprocess.CompletedProcess(command, 0, stdout="")
        return backends.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    backends.ensure_lima_instance()

    assert commands[0][0] == ["limactl", "list", "--format=json"]
    assert commands[1][0] == [
        "limactl",
        "start",
        "--tty=false",
        f"--name={backends.LIMA_INSTANCE}",
        "--vm-type=vz",
        "--arch=aarch64",
        "--rosetta",
        "--mount-writable",
        "template:apptainer",
    ]
    assert commands[1][1]["check"] is True


def test_ensure_lima_detects_python_running_under_rosetta(monkeypatch):
    commands = []
    monkeypatch.setattr(backends.shutil, "which", lambda executable: "/bin/limactl")
    monkeypatch.setattr(backends.platform, "machine", lambda: "x86_64")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["limactl", "list"]:
            return backends.subprocess.CompletedProcess(command, 0, stdout="")
        if command[0] == "sysctl":
            return backends.subprocess.CompletedProcess(command, 0, stdout="1\n")
        return backends.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    backends.ensure_lima_instance()

    start_command = next(command for command in commands if command[1] == "start")
    assert "--arch=aarch64" in start_command
    assert "--rosetta" in start_command


@pytest.mark.parametrize(
    ("status", "expected_starts"),
    [("Running", 0), ("Stopped", 1)],
)
def test_ensure_lima_reuses_existing_vm(monkeypatch, status, expected_starts):
    commands = []
    monkeypatch.setattr(backends.shutil, "which", lambda executable: "/bin/limactl")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["limactl", "list"]:
            output = '{"name":"vhmodels-apptainer","status":"' + status + '"}\n'
            return backends.subprocess.CompletedProcess(command, 0, stdout=output)
        return backends.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    backends.ensure_lima_instance()

    start_commands = [command for command in commands if command[1] == "start"]
    assert len(start_commands) == expected_starts
    if start_commands:
        assert start_commands[0] == [
            "limactl",
            "start",
            "--tty=false",
            backends.LIMA_INSTANCE,
        ]
