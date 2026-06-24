# Unit tests for the runtime backend abstraction (vhmodels.vh_checker.backends).
import pytest

from vhmodels.vh_checker import backends
from vhmodels.vh_checker.backends import CondaBackend, get_backend


# --- get_backend -----------------------------------------------------------


def test_get_backend_conda():
    backend = get_backend("conda", "vhmodels-dinobloom")
    assert isinstance(backend, CondaBackend)
    assert backend.env_name == "vhmodels-dinobloom"


def test_get_backend_apptainer_not_supported_yet():
    with pytest.raises(NotImplementedError, match="apptainer"):
        get_backend("apptainer", "vhmodels-dinobloom")


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
