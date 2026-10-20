"""Unit tests for the Ray runtime: model venvs, actor and options; none need Ray."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from vhmodels.models.registry import REGISTRY
from vhmodels.utils.uv_utils import UvSpec, model_uv_spec
from vhmodels.vh_checker import cli, factory
from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_ray import client, envs
from vhmodels.vh_ray.actor import ModelActor, hide_paths

_VERSIONS = {"python": "3.11.15", "ray": "2.59.0"}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VHMODELS_VENV_DIR", str(tmp_path / "venvs"))
    monkeypatch.delenv("VHMODELS_IMAGE_DIR", raising=False)
    monkeypatch.setattr(envs, "driver_versions", lambda: dict(_VERSIONS))


class FakeUv:
    """Record build commands and create the venv directory like uv would."""

    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[1] == "venv":
            Path(command[-1]).mkdir(parents=True)
            (Path(command[-1]) / "pyvenv.cfg").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


def test_build_commands_mirror_the_image_build(tmp_path):
    spec = model_uv_spec(REGISTRY.get_model("hyformer"))
    python = str(tmp_path / "bin" / "python")

    commands = envs.build_commands(spec, tmp_path, _VERSIONS, "Hyformer", "/repo")

    assert commands[0] == [
        "uv",
        "venv",
        "--managed-python",
        "--python",
        "3.11.15",
        str(tmp_path),
    ]
    assert commands[1] == [
        "uv", "pip", "install", "--python", python, "--compile-bytecode",
        "--torch-backend", "cu118",
        "--excludes", str(spec.exclude_path),
        "--requirements", str(spec.requirements_path),
        "ray==2.59.0",
    ]  # fmt: skip
    assert commands[2][-1] == "/repo"
    assert commands[3][-1].endswith("vhmodels.models.Hyformer.model")


def test_build_commands_without_spec_install_only_ray(tmp_path):
    commands = envs.build_commands(UvSpec(), tmp_path, _VERSIONS)

    assert commands[1][-2:] == ["--compile-bytecode", "ray==2.59.0"]
    assert commands[3][-1] == "import ray, vhmodels.vh_ray.actor"


def test_build_model_venv_reuses_and_rebuilds(monkeypatch, tmp_path):
    fake_uv = FakeUv()
    manifest = REGISTRY.get_model("prottrans")

    target = envs.build_model_venv("prottrans", manifest, run=fake_uv)
    envs.build_model_venv("prottrans", manifest, run=fake_uv)
    assert target == (tmp_path / "venvs" / "prottrans").resolve()
    assert len(fake_uv.commands) == 4
    # Ray runs the actor with the interpreter the requirements went into.
    install = fake_uv.commands[1]
    python = install[install.index("--python") + 1]
    assert envs.model_runtime_env("prottrans") == {"py_executable": python}

    monkeypatch.setattr(envs, "driver_versions", lambda: {**_VERSIONS, "ray": "2.60.0"})
    envs.build_model_venv("prottrans", manifest, run=fake_uv)
    assert len(fake_uv.commands) == 8
    assert "ray==2.60.0" in fake_uv.commands[5]


def test_build_model_venv_refuses_to_replace_foreign_directory(tmp_path):
    foreign = tmp_path / "venvs" / "prottrans"
    foreign.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="not a model venv"):
        envs.build_model_venv(
            "prottrans", REGISTRY.get_model("prottrans"), run=FakeUv()
        )


def test_check_model_venv_rejects_missing_and_outdated_venvs(monkeypatch, tmp_path):
    manifest = REGISTRY.get_model("prottrans")
    with pytest.raises(RuntimeError, match="No Ray venv"):
        envs.check_model_venv("prottrans", manifest)

    envs.build_model_venv("prottrans", manifest, run=FakeUv())
    envs.check_model_venv("prottrans", manifest)

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch\n", encoding="utf-8")
    edited = UvSpec(requirements_path=requirements)
    monkeypatch.setattr(envs, "model_uv_spec", lambda manifest: edited)
    with pytest.raises(RuntimeError, match="outdated.*requirements_sha256"):
        envs.check_model_venv("prottrans", manifest)


def test_create_ray_env_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(
        envs,
        "build_model_venv",
        lambda project, manifest, force: calls.append((project, force)),
    )

    result = CliRunner().invoke(cli.main, ["create-ray-env", "hyformer", "--force"])
    assert result.exit_code == 0, result.output
    assert calls == [("hyformer", True)]

    result = CliRunner().invoke(cli.main, ["create-ray-env", "unknown"])
    assert result.exit_code == 1
    assert "is not registered" in result.output


def test_load_model_checks_venv_and_resolves_on_the_host(monkeypatch):
    created, checked = [], []
    resources = {"weights": {"type": "huggingface", "repo_id": "r", "files": {}}}
    monkeypatch.setattr(
        client, "RayModel", lambda **kwargs: created.append(kwargs) or "ray-model"
    )
    monkeypatch.setattr(
        envs, "check_model_venv", lambda project, manifest: checked.append(project)
    )
    monkeypatch.setattr(
        factory, "_resolve_resources", lambda project, model: (model, resources)
    )

    model = factory.load_model(
        "hyformer",
        model="hyformer_molecules_8M",
        runtime="ray",
        ray_options={"num_cpus": 2},
    )

    assert model == "ray-model"
    assert checked == ["hyformer"]
    assert created == [
        {
            "project": "hyformer",
            "model": "hyformer_molecules_8M",
            "resources": resources,
            "load_kwargs": {},
            "ray_options": {"num_cpus": 2},
            "image_path": None,
        }
    ]


def test_load_model_rejects_ray_options_for_other_runtimes():
    with pytest.raises(ValueError, match="runtime='ray'"):
        factory.load_model("hyformer", runtime="conda", ray_options={"num_gpus": 1})


def test_hide_paths_removes_driver_directories(monkeypatch, tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    (tmp_path / "link").symlink_to(script_dir)
    monkeypatch.setattr(
        sys, "path", [str(tmp_path / "link"), "/venv/lib", str(tmp_path)]
    )

    hide_paths([str(script_dir), str(tmp_path)])

    assert sys.path == ["/venv/lib"]


@pytest.mark.parametrize(
    ("device", "num_gpus"), [("cuda:1", 1), ("auto", 0)]
)
def test_actor_options_request_a_gpu_for_cuda_devices(device, num_gpus):
    options = client._actor_options("prottrans", {"device": device})

    assert options["num_gpus"] == num_gpus
    assert options["runtime_env"] == envs.model_runtime_env("prottrans")


@pytest.mark.parametrize(("device", "nv"), [("cuda:0", ["--nv"]), ("cpu", [])])
def test_actor_options_run_the_actor_in_an_image(device, nv):
    options = client._actor_options("mole", {"device": device}, None, "/i/m.sif")

    assert options["runtime_env"] == {
        "py_executable": " ".join(
            ["apptainer", "exec", *nv, "/i/m.sif", "/opt/venv/bin/python"]
        )
    }


def test_load_model_with_image_skips_the_venv(monkeypatch, tmp_path):
    created = []
    image = tmp_path / "mole.sif"
    monkeypatch.setattr(client, "RayModel", lambda **kwargs: created.append(kwargs))
    monkeypatch.setattr(envs, "check_model_venv", pytest.fail)
    monkeypatch.setattr(factory, "_resolve_resources", lambda project, model: (model, {}))

    with pytest.raises(RuntimeError, match="does not exist"):
        factory.load_model("mole", runtime="ray", image_path=image)
    image.touch()
    factory.load_model("mole", runtime="ray", image_path=image)
    monkeypatch.setenv("VHMODELS_IMAGE_DIR", str(tmp_path))
    (tmp_path / "vhmodels-mole.sif").touch()
    factory.load_model("mole", runtime="ray")

    assert [kwargs["image_path"] for kwargs in created] == [
        image,
        tmp_path / "vhmodels-mole.sif",
    ]


def test_ray_options_override_actor_defaults():
    options = client._actor_options(
        "prottrans", {"device": "cuda"}, {"num_gpus": 2, "num_cpus": 4}
    )

    assert options["num_gpus"] == 2
    assert options["num_cpus"] == 4


def _persistent_fake():
    path = Path(__file__).parent / "fixtures" / "persistent_worker" / "model.py"
    spec = importlib.util.spec_from_file_location("persistent_fake_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PersistentFake


def test_model_actor_serves_the_model_like_a_worker(monkeypatch, tmp_path):
    # ModelActor is a plain class, so the real fixture model runs in-process.
    monkeypatch.setattr(
        BaseModel, "get_class", staticmethod(lambda project: _persistent_fake())
    )
    driver_dir = tmp_path / "driver"
    driver_dir.mkdir()
    monkeypatch.setattr(sys, "path", [str(driver_dir), *sys.path])
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    caller_cwd = os.getcwd()

    actor = ModelActor(
        "persistent-fake",
        "tiny-variant",
        {"weights": {"type": "local", "path": "/weights"}},
        {"device": "cpu"},
        [str(driver_dir)],
    )
    first = actor.embed(["a"], cwd=str(request_dir), batch_size=2)["output"]
    second = actor.embed(["b"])["output"]

    assert str(driver_dir) not in sys.path
    assert first["model"] == "tiny-variant"
    assert first["resources"] == {"weights": {"type": "local", "path": "/weights"}}
    assert first["load_kwargs"] == {"device": "cpu"}
    assert first["kwargs"] == {"batch_size": 2}
    assert os.path.realpath(first["cwd"]) == os.path.realpath(request_dir)
    assert second["cwd"] == os.getcwd() == caller_cwd
    assert first["load_id"] == second["load_id"]
    assert [first["embed_count"], second["embed_count"]] == [1, 2]
    # PersistentFake has no predict(); the model's own error surfaces.
    with pytest.raises(AttributeError, match="predict"):
        actor.predict("anything")
