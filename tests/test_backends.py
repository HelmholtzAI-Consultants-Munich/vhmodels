# Unit tests for the runtime backend abstraction (vhmodels.vh_checker.backends).
import pytest

from vhmodels.utils import lima_utils
from vhmodels.vh_checker import backends
from vhmodels.vh_checker.backends import ApptainerBackend, CondaBackend, get_backend


# --- get_backend -----------------------------------------------------------


def test_get_backend_selects_runtime():
    conda = get_backend("conda", "vhmodels-dinobloom")
    assert isinstance(conda, CondaBackend)
    assert conda.env_name == "vhmodels-dinobloom"
    apptainer = get_backend("apptainer", "/images/vhmodels-dinobloom.sif")
    assert isinstance(apptainer, ApptainerBackend)
    assert apptainer.image_path == "/images/vhmodels-dinobloom.sif"
    with pytest.raises(NotImplementedError):
        get_backend("docker", "vhmodels-dinobloom")


# --- CondaBackend ----------------------------------------------------------


def test_conda_build_worker_start_command():
    backend = CondaBackend("vhmodels-prottrans")
    cmd = backend.build_worker_start_command("/tmp/model-worker.sock")
    assert cmd == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "vhmodels-prottrans",
        "python",
        "-m",
        "vhmodels.vh_checker.worker",
        "serve",
        "/tmp/model-worker.sock",
    ]


@pytest.mark.parametrize(
    ("environments", "available"),
    [("base\nvhmodels-mole\n", True), ("base\n", False)],
)
def test_conda_is_available(monkeypatch, environments, available):
    class FakeResult:
        stdout = "# conda environments:\n" + environments

    monkeypatch.setattr(backends.subprocess, "run", lambda *a, **k: FakeResult())
    assert CondaBackend("vhmodels-mole").is_available() is available


def test_conda_subprocess_env_strips_only_pythonpath(monkeypatch):
    # vhmodels is installed into the env, so PYTHONPATH is unnecessary and a
    # host entry could shadow the env's packages -- it must be removed.
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env = CondaBackend("vhmodels-mole").subprocess_env()
    assert "PYTHONPATH" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


# --- ApptainerBackend ------------------------------------------------------


def test_apptainer_is_available_only_for_existing_file(tmp_path):
    image_path = tmp_path / "vhmodels-mole.sif"
    image_path.touch()
    directory_path = tmp_path / "directory.sif"
    directory_path.mkdir()
    assert ApptainerBackend(image_path).is_available() is True
    assert ApptainerBackend(tmp_path / "missing.sif").is_available() is False
    assert ApptainerBackend(directory_path).is_available() is False


def test_apptainer_subprocess_env_strips_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    monkeypatch.setenv("APPTAINERENV_PYTHONPATH", "/override/apptainer")
    monkeypatch.setenv("APPTAINER_BINDPATH", "/data")
    env = ApptainerBackend("vhmodels-mole.sif", use_lima=False).subprocess_env()
    assert "PYTHONPATH" not in env
    assert "APPTAINERENV_PYTHONPATH" not in env
    assert env["APPTAINER_BINDPATH"] == "/data"


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
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: True)
    monkeypatch.setattr(
        lima_utils,
        "ensure_lima_instance",
        lambda: None,
    )

    backend = ApptainerBackend("model.sif", use_lima=True)
    backend.prepare()
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: False)

    with pytest.raises(RuntimeError, match="relative input paths"):
        backend.validate_request_cwd("/unshared/workdir")


def test_ensure_lima_creates_arm64_rosetta_vm_even_from_rosetta_python(monkeypatch):
    commands = []
    monkeypatch.setattr(lima_utils.shutil, "which", lambda executable: "/bin/limactl")
    # An x86_64 Python running under Rosetta on Apple Silicon.
    monkeypatch.setattr(lima_utils.platform, "machine", lambda: "x86_64")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if command[:2] == ["limactl", "list"]:
            return lima_utils.subprocess.CompletedProcess(command, 0, stdout="")
        if command[0] == "sysctl":
            return lima_utils.subprocess.CompletedProcess(command, 0, stdout="1\n")
        return lima_utils.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lima_utils.subprocess, "run", fake_run)

    lima_utils.ensure_lima_instance()

    starts = [(command, kwargs) for command, kwargs in commands if command[1] == "start"]
    assert starts == [
        (
            [
                "limactl",
                "start",
                "--tty=false",
                f"--name={lima_utils.LIMA_INSTANCE}",
                "--vm-type=vz",
                "--arch=aarch64",
                "--rosetta",
                "--mount-writable",
                "template:apptainer",
            ],
            {"check": True},
        )
    ]


@pytest.mark.parametrize(
    ("status", "expected_starts"),
    [("Running", 0), ("Stopped", 1)],
)
def test_ensure_lima_reuses_existing_vm(monkeypatch, status, expected_starts):
    commands = []
    monkeypatch.setattr(lima_utils.shutil, "which", lambda executable: "/bin/limactl")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["limactl", "list"]:
            output = '{"name":"vhmodels-apptainer","status":"' + status + '"}\n'
            return lima_utils.subprocess.CompletedProcess(command, 0, stdout=output)
        return lima_utils.subprocess.CompletedProcess(command, 0)

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


def test_apptainer_instance_commands_are_lima_wrapped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    backend = ApptainerBackend("/images/model.sif", use_lima=True)

    commands = [
        backend.build_instance_start_command("worker-1", "/tmp/worker.sock"),
        backend.build_instance_request_command("worker-1", "/tmp/worker.sock", 30),
        backend.build_instance_stop_command("worker-1"),
        backend.build_instance_list_command("worker-1"),
    ]

    for command in commands:
        assert command[:7] == [
            "limactl",
            "shell",
            "--tty=false",
            "--preserve-env",
            "--workdir",
            str(backends.Path.home().resolve()),
            lima_utils.LIMA_INSTANCE,
        ]

