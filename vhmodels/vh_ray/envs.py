"""Per-model uv venvs for the Ray runtime and the runtime_env that selects them."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path

from vhmodels.utils.uv_utils import MODEL_PYTHON, model_uv_spec, uv_build_cache_dir

VENV_MARKER = "vhmodels-ray-env.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def venv_dir(project):
    """Return ``$VHMODELS_VENV_DIR/<project>`` (default root: ~/.cache/vhmodels/venvs)."""
    root = os.environ.get("VHMODELS_VENV_DIR")
    if not root:
        cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        root = Path(cache_home) / "vhmodels" / "venvs"
    return Path(root).expanduser().resolve() / project


def driver_versions():
    """Return the Python and Ray versions every model venv must match."""
    try:
        ray_version = importlib.metadata.version("ray")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("Ray is not installed; install vhmodels[ray].") from None
    return {"python": platform.python_version(), "ray": ray_version}


def build_commands(spec, target, versions, model_dir_name=None, repo_root=None):
    """Return the uv commands that build one model venv (mirrors the Apptainer %post)."""
    python = str(Path(target) / "bin" / "python")
    install = ["uv", "pip", "install", "--python", python, "--compile-bytecode"]
    if spec.torch_backend:
        install += ["--torch-backend", spec.torch_backend]
    if spec.exclude_path:
        install += ["--excludes", str(spec.exclude_path)]
    if spec.requirements_path:
        install += ["--requirements", str(spec.requirements_path)]
    install.append(f"ray[serve]=={versions['ray']}")

    smoke_import = "import ray, vhmodels.vh_ray.actor"
    if model_dir_name:
        smoke_import += f", vhmodels.models.{model_dir_name}.model"

    editable = ["uv", "pip", "install", "--python", python, "--no-deps", "--editable"]
    return [
        ["uv", "venv", "--managed-python", "--python", versions["python"], str(target)],
        install,
        editable + [str(repo_root or REPO_ROOT)],
        [python, "-I", "-c", smoke_import],
    ]


def _model_dir_name(manifest):
    return Path(manifest.model_dir).name if manifest.model_dir else None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def _fingerprint(manifest, repo_root=None):
    """Describe everything a venv is built from; any difference means rebuild."""
    spec = model_uv_spec(manifest)
    versions = driver_versions()
    commands = build_commands(
        spec, "<venv>", versions, _model_dir_name(manifest), repo_root
    )
    return {
        **versions,
        "requirements_sha256": _sha256(spec.requirements_path),
        "exclude_sha256": _sha256(spec.exclude_path),
        "commands": commands,
    }


def _read_marker(target):
    try:
        return json.loads((target / VENV_MARKER).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_model_venv(
    project, manifest, force=False, repo_root=None, run=subprocess.run
):
    """Build the venv a model's Ray actor runs in, unless an up-to-date one exists."""
    spec = model_uv_spec(manifest)
    for path in (spec.requirements_path, spec.exclude_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"'{project}' dependency file not found: {path}")

    fingerprint = _fingerprint(manifest, repo_root)
    if fingerprint["python"] != MODEL_PYTHON:
        warnings.warn(
            f"Model environments use Python {MODEL_PYTHON}; "
            f"building '{project}' with the driver's {fingerprint['python']}."
        )

    target = venv_dir(project)
    if not force and _read_marker(target) == fingerprint:
        print(f"Ray venv for '{project}' is up to date at '{target}'.")
        return target

    if target.exists():
        if not ((target / VENV_MARKER).is_file() or (target / "pyvenv.cfg").is_file()):
            raise RuntimeError(f"Refusing to replace '{target}': not a model venv.")
        shutil.rmtree(target)

    print(f"Building Ray venv for '{project}' at '{target}'.")
    environment = dict(os.environ, UV_CACHE_DIR=str(uv_build_cache_dir()))
    commands = build_commands(
        spec, target, fingerprint, _model_dir_name(manifest), repo_root
    )
    for command in commands:
        run(command, check=True, env=environment)

    marker = json.dumps(fingerprint, indent=2) + "\n"
    (target / VENV_MARKER).write_text(marker, encoding="utf-8")
    return target


def check_model_venv(project, manifest, repo_root=None):
    """Raise unless the model's venv was built from the current spec and driver."""
    marker = _read_marker(venv_dir(project))
    hint = f"Run 'vh-checker create-ray-env {project}'."
    if marker is None:
        raise RuntimeError(f"No Ray venv for '{project}'. {hint}")
    expected = _fingerprint(manifest, repo_root)
    outdated = [key for key in expected if marker.get(key) != expected[key]]
    if outdated:
        raise RuntimeError(
            f"Ray venv for '{project}' is outdated ({', '.join(outdated)}). {hint}"
        )


def image_dir_path(project):
    """Return the model's image in ``$VHMODELS_IMAGE_DIR``, or None if that is unset."""
    root = os.environ.get("VHMODELS_IMAGE_DIR")
    if not root:
        return None
    return Path(root).expanduser().resolve() / f"vhmodels-{project}.sif"


def model_runtime_env(project):
    """Return the Ray runtime_env that runs an actor in the model's venv."""
    return {"py_executable": str(venv_dir(project) / "bin" / "python")}


def image_runtime_env(image_path, gpu=False):
    """Return the Ray runtime_env that runs an actor in the model's Apptainer image."""
    command = ["apptainer", "exec", *(["--nv"] if gpu else [])]
    command += [str(image_path), "/opt/venv/bin/python"]
    return {"py_executable": shlex.join(command)}
