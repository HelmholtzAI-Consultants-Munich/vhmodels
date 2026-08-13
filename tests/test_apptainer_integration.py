"""End-to-end tests for the persistent Apptainer model process manager.

The test builds a small SIF with no model dependencies or weights. It is opt-in
because building an image requires an external runtime and may pull a base
image. The same test uses Lima on macOS and native Apptainer on Linux.
"""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import pytest

from vhmodels.utils import lima_utils
from vhmodels.vh_checker import factory


_RUN_INTEGRATION = os.environ.get("VHMODELS_RUN_APPTAINER_INTEGRATION") == "1"
_PROJECT = "persistent-fake"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vhmodels"
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "apptainer_tiny"
_IS_MACOS = platform.system() == "Darwin"
_APPTAINER_OPTION_ENV = (
    "APPTAINER_BIND",
    "APPTAINER_BINDPATH",
    "APPTAINER_NV",
    "APPTAINER_ROCM",
    "APPTAINERENV_PYTHONPATH",
    "PYTHONPATH",
    "SINGULARITY_BIND",
    "SINGULARITY_BINDPATH",
    "SINGULARITY_NV",
    "SINGULARITY_ROCM",
    "SINGULARITYENV_PYTHONPATH",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _RUN_INTEGRATION,
        reason=(
            "set VHMODELS_RUN_APPTAINER_INTEGRATION=1 to build and run the "
            "tiny Apptainer lifecycle test"
        ),
    ),
]


def _runtime_environment():
    """Return a CPU-only environment without caller bind-path overrides."""
    environment = os.environ.copy()
    for key in _APPTAINER_OPTION_ENV:
        environment.pop(key, None)
    return environment


def _prepare_runtime():
    system = platform.system()
    if system == "Darwin":
        if shutil.which("limactl") is None:
            pytest.skip("Lima is not installed on this macOS host.")
        lima_utils.ensure_lima_instance()
        return
    if system == "Linux":
        if shutil.which("apptainer") is None:
            pytest.skip("Apptainer is not installed on this Linux host.")
        return
    pytest.skip(f"Apptainer integration is not supported on {system}.")


def _stage_build_context(build_context):
    shutil.copytree(
        _PACKAGE_ROOT,
        build_context / "vhmodels",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        _FIXTURE_ROOT / "PersistentFake",
        build_context / "vhmodels" / "models" / "PersistentFake",
    )
    shutil.copy2(_FIXTURE_ROOT / "Apptainer.def", build_context / "Apptainer.def")


def _build_image(image_path, build_context):
    definition_path = build_context / "Apptainer.def"
    command = ["apptainer", "build", str(image_path), str(definition_path)]
    if _IS_MACOS:
        command = lima_utils.lima_shell_command(
            [
                "env",
                "APPTAINER_TMPDIR=/var/tmp",
                "apptainer",
                "build",
                "--arch",
                "amd64",
                str(image_path),
                str(definition_path),
            ],
            build_context,
        )
    subprocess.run(
        command,
        check=True,
        cwd=build_context,
        env=_runtime_environment(),
    )


@pytest.fixture(scope="session")
def tiny_apptainer_image():
    """Build a temporary dependency-free SIF for this test session."""
    _prepare_runtime()

    temporary_parent = Path.home() if _IS_MACOS else None
    with tempfile.TemporaryDirectory(
        prefix="vhmodels-apptainer-integration-",
        dir=temporary_parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        build_context = temporary_path / "build"
        build_context.mkdir()
        _stage_build_context(build_context)
        image_path = temporary_path / "persistent-fake.sif"
        _build_image(image_path, build_context)
        assert image_path.is_file()
        yield image_path


@pytest.fixture
def shared_runtime_directory():
    """Create a host directory that is also visible inside Lima/Apptainer."""
    with tempfile.TemporaryDirectory(
        prefix="vhmodels-apptainer-runtime-",
        dir=Path.home(),
    ) as directory:
        yield Path(directory)


def _run_instance_command(backend, command):
    return subprocess.run(
        backend._wrap_command(command),
        capture_output=True,
        text=True,
        env=backend.subprocess_env(),
        timeout=30,
    )


def _instance_is_registered(backend, instance_name):
    result = _run_instance_command(
        backend,
        ["apptainer", "instance", "list", instance_name],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect Apptainer instance '{instance_name}'.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return any(
        columns and columns[0] == instance_name
        for columns in (line.split() for line in result.stdout.splitlines())
    )


def _stop_exact_instance(backend, instance_name):
    return _run_instance_command(
        backend,
        ["apptainer", "instance", "stop", instance_name],
    )


def test_persistent_worker_reuse_cleanup_and_restart(
    tiny_apptainer_image,
    shared_runtime_directory,
    monkeypatch,
):
    """Exercise the real SIF, instance lifecycle, socket relay, and model worker."""
    for key in _APPTAINER_OPTION_ENV:
        monkeypatch.delenv(key, raising=False)
    # Some HPC configurations disable Apptainer's default home bind. Make the
    # test's two host-visible files explicit before the instance starts.
    monkeypatch.setenv("APPTAINER_BINDPATH", str(shared_runtime_directory))
    monkeypatch.chdir(shared_runtime_directory)
    monkeypatch.setitem(
        factory.MODEL_REGISTRY,
        _PROJECT,
        {"class_path": "PersistentFake.model.PersistentFake"},
    )

    model = factory.load_model(
        project=_PROJECT,
        model="tiny-variant",
        runtime="apptainer",
        image_path=tiny_apptainer_image,
        sentinel="forwarded-load-kwarg",
    )
    manager = model._process_manager
    manager.timeout = 60.0
    backend = model.backend
    # Track the generated identity before any subprocess can fail so teardown
    # can still target an instance left behind by initial startup/load failure.
    owned_instance_names = [manager.instance_name]

    try:
        with model:
            first = model.embed(
                {"records": [1, None, {"unicode": "München"}]},
                batch_size=2,
            )
            first_instance_name = manager.instance_name
            second = model.embed(["second", {"nested": True}], normalize=True)

            assert first["load_id"] == second["load_id"]
            assert first["pid"] == second["pid"]
            assert [first["embed_count"], second["embed_count"]] == [1, 2]
            assert first["model"] == "tiny-variant"
            assert first["load_kwargs"] == {"sentinel": "forwarded-load-kwarg"}
            assert first["kwargs"] == {"batch_size": 2}
            assert second["kwargs"] == {"normalize": True}
            assert _instance_is_registered(backend, first_instance_name)

            assert _stop_exact_instance(backend, first_instance_name).returncode == 0
            with pytest.raises(RuntimeError, match="Model subprocess failed"):
                model.embed({"after": "external stop"})

            assert not manager.is_started
            assert not _instance_is_registered(backend, first_instance_name)
            restarted_instance_name = manager.instance_name
            assert restarted_instance_name != first_instance_name
            owned_instance_names.append(restarted_instance_name)

            after_restart = model.embed({"after": "transport failure"})
            assert after_restart["load_id"] != first["load_id"]
            assert after_restart["embed_count"] == 1
            assert _instance_is_registered(backend, restarted_instance_name)

        assert not _instance_is_registered(backend, restarted_instance_name)
    finally:
        try:
            model.close()
        except Exception:
            pass
        # Assertion failures must not leak user-owned instances or GPU memory.
        for instance_name in dict.fromkeys(owned_instance_names):
            try:
                _stop_exact_instance(backend, instance_name)
            except (OSError, subprocess.SubprocessError):
                pass
