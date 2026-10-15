"""Opt-in end-to-end test: PersistentFake in a real uv venv via load_model(runtime="ray")."""

import os
from pathlib import Path
import shutil

import pytest

from vhmodels.models.registry import Registry
from vhmodels.vh_checker import factory


_RUN_INTEGRATION = os.environ.get("VHMODELS_RUN_RAY_INTEGRATION") == "1"
_PROJECT = "persistent-fake"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "persistent_worker"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _RUN_INTEGRATION,
        reason="set VHMODELS_RUN_RAY_INTEGRATION=1 to build and run the Ray test",
    ),
]


@pytest.fixture(scope="module")
def staged_checkout(tmp_path_factory):
    """Copy vhmodels plus PersistentFake into an installable source tree."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed on this host.")
    staging_root = tmp_path_factory.mktemp("vhmodels-ray-source")
    shutil.copy2(_REPOSITORY_ROOT / "pyproject.toml", staging_root)
    shutil.copytree(
        _REPOSITORY_ROOT / "vhmodels",
        staging_root / "vhmodels",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        _FIXTURE_ROOT,
        staging_root / "vhmodels" / "models" / "PersistentFake",
        ignore=shutil.ignore_patterns("apptainer.def"),
    )
    return staging_root


@pytest.fixture(scope="module")
def ray_cluster(tmp_path_factory):
    ray = pytest.importorskip("ray")
    # Start outside this checkout so workers import the staged vhmodels copy.
    previous_cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("ray-driver-cwd"))
    try:
        ray.init(num_cpus=2, include_dashboard=False)
        yield ray
        ray.shutdown()
    finally:
        os.chdir(previous_cwd)


def test_ray_actor_runs_in_model_venv(
    staged_checkout, ray_cluster, monkeypatch, tmp_path
):
    from vhmodels.vh_ray import envs

    registry = Registry(models_dir=staged_checkout / "vhmodels" / "models")
    monkeypatch.setenv("VHMODELS_VENV_DIR", str(tmp_path / "venvs"))
    monkeypatch.setattr(envs, "REPO_ROOT", staged_checkout)
    monkeypatch.setattr(factory, "_registry", registry)
    venv = envs.build_model_venv(_PROJECT, registry.get_model(_PROJECT))
    monkeypatch.chdir(tmp_path)

    model = factory.load_model(
        _PROJECT,
        model="tiny-variant",
        runtime="ray",
        sentinel="forwarded-load-kwarg",
    )
    with model:
        info = model.info()
        first = model.embed(
            {"records": [1, None, {"unicode": "München"}]}, batch_size=2
        )
        second = model.embed(["second"], normalize=True)

        assert Path(info["executable"]) == venv / "bin" / "python"
        # The driver's directories are hidden from the model's imports.
        assert os.path.realpath(os.getcwd()) not in map(
            os.path.realpath, info["sys_path"]
        )
        assert first["pid"] == second["pid"] == info["pid"] != os.getpid()
        assert first["load_id"] == second["load_id"]
        assert [first["embed_count"], second["embed_count"]] == [1, 2]
        assert first["model"] == "tiny-variant"
        assert first["resources"] == {}
        assert first["load_kwargs"] == {"sentinel": "forwarded-load-kwarg"}
        assert first["kwargs"] == {"batch_size": 2}
        assert second["input"] == ["second"]

        # PersistentFake has no predict(); the model's error reaches the host.
        with pytest.raises(AttributeError, match="predict"):
            model.predict("anything")

    with pytest.raises(RuntimeError, match="closed"):
        model.embed("after close")
