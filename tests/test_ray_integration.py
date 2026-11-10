"""Opt-in end-to-end tests: PersistentFake in a real uv venv via Ray Core and Ray Serve."""

import json
import os
from pathlib import Path
import shutil
import socket
import urllib.error
import urllib.request

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


@pytest.fixture(scope="module")
def model_venv(staged_checkout, tmp_path_factory):
    """Build PersistentFake's venv once and resolve the model from the staged copy."""
    from vhmodels.vh_ray import envs

    registry = Registry(models_dir=staged_checkout / "vhmodels" / "models")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VHMODELS_VENV_DIR", str(tmp_path_factory.mktemp("venvs")))
        patch.setattr(envs, "REPO_ROOT", staged_checkout)
        patch.setattr(factory, "_registry", registry)
        yield envs.build_model_venv(_PROJECT, registry.get_model(_PROJECT))


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ray_actor_runs_in_model_venv(model_venv, ray_cluster, monkeypatch, tmp_path):
    venv = model_venv
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
        # Relative input paths resolve against the caller's directory.
        assert os.path.realpath(first["cwd"]) == os.path.realpath(tmp_path)

        # PersistentFake has no predict(); the model's error reaches the host.
        with pytest.raises(AttributeError, match="predict"):
            model.predict("anything")

    with pytest.raises(RuntimeError, match="closed"):
        model.embed("after close")


def test_serve_app_runs_in_model_venv(model_venv, ray_cluster, monkeypatch):
    import requests
    from ray import serve

    from vhmodels.models.schema import ModelInput
    from vhmodels.vh_ray.serve.deployment import build_app

    # Image-style input, so uploaded files reach the model as a list of paths.
    monkeypatch.setattr(
        factory._get_registry().get_model(_PROJECT).implementation,
        "input",
        ModelInput(type="images"),
    )
    port = _free_port()
    serve.start(http_options={"host": "127.0.0.1", "port": port})
    try:
        app = build_app(
            {
                "project": _PROJECT,
                "model": "tiny-variant",
                "ray_options": {"num_cpus": 0.5},
                "sentinel": "forwarded-load-kwarg",
            }
        )
        serve.run(app, name=_PROJECT, route_prefix=f"/{_PROJECT}")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/{_PROJECT}/embed",
            data=json.dumps({"input": ["over-http"], "batch_size": 2}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            over_http = json.load(response)["output"]
        # Input of the wrong shape is rejected before the model runs.
        invalid = urllib.request.Request(
            f"http://127.0.0.1:{port}/{_PROJECT}/embed",
            data=json.dumps({"input": 5}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(invalid, timeout=60)
        handle = serve.get_app_handle(_PROJECT)
        info = handle.info.remote().result()
        over_handle = handle.embed.remote(["over-handle"]).result()["output"]
        uploaded = requests.post(
            f"http://127.0.0.1:{port}/{_PROJECT}/embed/files",
            files=[("input", ("b.tif", b"b")), ("input", ("../a.tif", b"a"))],
            data={"kwargs": json.dumps({"batch_size": 2})},
            timeout=60,
        )
        no_files = requests.post(
            f"http://127.0.0.1:{port}/{_PROJECT}/embed/files",
            data={"kwargs": "{}"},
            timeout=60,
        )
    finally:
        serve.shutdown()

    assert Path(info["executable"]) == model_venv / "bin" / "python"
    assert over_http["input"] == ["over-http"]
    assert over_http["kwargs"] == {"batch_size": 2}
    assert over_http["load_kwargs"] == {"sentinel": "forwarded-load-kwarg"}
    assert rejected.value.code == 422
    assert over_handle["input"] == ["over-handle"]
    # HTTP and handle calls reach the same replica and loaded model.
    assert over_http["pid"] == over_handle["pid"] == info["pid"]
    assert over_http["load_id"] == over_handle["load_id"]
    assert [over_http["embed_count"], over_handle["embed_count"]] == [1, 2]
    # Uploads reach the model as paths, in order and without directories.
    assert uploaded.status_code == 200, uploaded.text
    over_upload = uploaded.json()["output"]
    assert [Path(path).name for path in over_upload["input"]] == ["b.tif", "a.tif"]
    assert over_upload["kwargs"] == {"batch_size": 2}
    assert over_upload["embed_count"] == 3
    # The files exist only while the request runs.
    assert not Path(over_upload["input"][0]).parent.parent.exists()
    assert no_files.status_code == 422
