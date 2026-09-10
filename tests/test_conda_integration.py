"""End-to-end test for the persistent Conda model process manager.

The test creates a temporary dependency-free Conda environment and stages the
current vhmodels checkout plus PersistentFake into it. It is opt-in because
creating the environment may require downloading a Python package.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import pytest

from vhmodels.vh_checker import factory


_RUN_INTEGRATION = os.environ.get("VHMODELS_RUN_CONDA_INTEGRATION") == "1"
_PROJECT = "persistent-fake"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vhmodels"
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "persistent_worker"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _RUN_INTEGRATION,
        reason=(
            "set VHMODELS_RUN_CONDA_INTEGRATION=1 to create and run the "
            "tiny Conda lifecycle test"
        ),
    ),
]


def _run(command, *, environment, cwd=None, timeout=300):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=cwd,
        timeout=timeout,
    )


def _stage_package(staging_root):
    shutil.copytree(
        _PACKAGE_ROOT,
        staging_root / "vhmodels",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        _FIXTURE_ROOT,
        staging_root / "vhmodels" / "models" / "PersistentFake",
        ignore=shutil.ignore_patterns("apptainer.def"),
    )


@pytest.fixture(scope="session")
def tiny_conda_environment():
    """Create an isolated Conda environment containing the staged package."""
    if shutil.which("conda") is None:
        pytest.skip("Conda is not installed on this host.")

    with tempfile.TemporaryDirectory(
        prefix="vhmodels-conda-integration-"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        conda_envs_path = temporary_path / "envs"
        staging_root = temporary_path / "source"
        staging_root.mkdir()
        _stage_package(staging_root)

        environment = os.environ.copy()
        environment["CONDA_ENVS_PATH"] = str(conda_envs_path)
        environment.pop("PYTHONPATH", None)
        env_name = f"vhmodels-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _run(
            [
                "conda",
                "create",
                "--yes",
                "--no-default-packages",
                "--name",
                env_name,
                f"python={python_version}",
            ],
            environment=environment,
        )

        site_result = _run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "--name",
                env_name,
                "python",
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            environment=environment,
            cwd=temporary_path,
        )
        site_packages = Path(site_result.stdout.strip().splitlines()[-1])
        (site_packages / "vhmodels-integration.pth").write_text(
            str(staging_root) + "\n",
            encoding="utf-8",
        )

        yield env_name, conda_envs_path, temporary_path


def test_persistent_conda_worker_reuse_cleanup_and_restart(
    tiny_conda_environment,
    monkeypatch,
):
    """Exercise real conda-run startup, sockets, reuse, restart, and cleanup."""
    env_name, conda_envs_path, runtime_directory = tiny_conda_environment
    monkeypatch.setenv("CONDA_ENVS_PATH", str(conda_envs_path))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    # Keep the worker's startup cwd outside this checkout so the staged package
    # in the temporary environment is the one it imports.
    monkeypatch.chdir(runtime_directory)

    model = factory.ModelProxy(
        project=_PROJECT,
        env_name=env_name,
        model="tiny-variant",
        runtime="conda",
        load_kwargs={"sentinel": "forwarded-load-kwarg"},
    )
    manager = model._process_manager
    manager.timeout = 60.0

    restarted_process = None
    restarted_socket = None
    try:
        with model:
            first = model.embed(
                {"records": [1, None, {"unicode": "München"}]},
                batch_size=2,
            )
            first_process = manager.transport._process
            first_socket = Path(manager.socket_path)
            second = model.embed(["second", {"nested": True}], normalize=True)

            assert first["load_id"] == second["load_id"]
            assert first["pid"] == second["pid"]
            assert [first["embed_count"], second["embed_count"]] == [1, 2]
            assert first["model"] == "tiny-variant"
            assert first["load_kwargs"] == {"sentinel": "forwarded-load-kwarg"}
            assert first["kwargs"] == {"batch_size": 2}
            assert second["kwargs"] == {"normalize": True}
            assert first_process.poll() is None
            assert first_socket.is_socket()

            manager.timeout = 0.25
            with pytest.raises(RuntimeError, match="request exceeded"):
                model.embed({"sleep": 30})
            manager.timeout = 60.0

            assert not manager.is_started
            assert first_process.poll() is not None
            assert not first_socket.exists()

            after_restart = model.embed({"after": "transport failure"})
            restarted_process = manager.transport._process
            restarted_socket = Path(manager.socket_path)
            assert after_restart["load_id"] != first["load_id"]
            assert after_restart["embed_count"] == 1
            assert restarted_process.poll() is None
            assert restarted_socket.is_socket()

        assert restarted_process.poll() is not None
        assert not restarted_socket.exists()
    finally:
        model.close()
