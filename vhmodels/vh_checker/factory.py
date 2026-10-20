"""Host-side model proxy and subprocess handover protocol."""

import json
from pathlib import Path

from vhmodels.vh_checker.backends import get_backend
from vhmodels.vh_checker.process_manager import (
    ApptainerProcessManager,
    CondaProcessManager,
)
from vhmodels.vh_checker.protocol import RESULT_MARKER
from vhmodels.utils.subprocess_utils import run_subprocess as _run_subprocess

DEFAULT_TIMEOUT = 600

# vhmodels/__init__.py imports this module to expose load_model(), and that
# import runs inside every isolated model worker (including dependency-free
# test fixtures with no third-party packages at all -- see
# tests/fixtures/persistent_worker). vhmodels.models.registry needs Pydantic,
# so it must only be imported lazily, once load_model() actually runs on the
# host. Tests override the cache directly, e.g.
# monkeypatch.setattr(factory, "_registry", Registry(models_dir=...)).
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from vhmodels.models.registry import Registry

        _registry = Registry()
    return _registry


def _resolve_resources(project, model):
    """Resolve a variant's manifest and sources on the host (may download files)."""
    return _get_registry().resolve_resources(project, model)


def _extract_frame(stdout, stderr):
    """Extract and decode one RESULT_MARKER-framed JSON value."""

    def _fail(reason):
        raise ValueError(
            f"{reason}\n"
            f"--- subprocess stdout ---\n{stdout}\n"
            f"--- subprocess stderr ---\n{stderr}"
        )

    open_idx = stdout.find(RESULT_MARKER)
    if open_idx == -1:
        _fail("No result marker found in subprocess output (no opening marker).")

    start = open_idx + len(RESULT_MARKER)
    # Use the final marker so model data may itself contain the marker string.
    close_idx = stdout.rfind(RESULT_MARKER)
    if close_idx < start:
        _fail("Result truncated: opening marker present but closing marker missing.")

    chunk = stdout[start:close_idx]
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError as e:
        _fail(f"Result frame is not valid JSON ({e}).")

    return parsed


def _unwrap_model_result(parsed, stdout="", stderr=""):
    """Return the value under the model's established ``output`` envelope."""
    if not isinstance(parsed, dict) or "output" not in parsed:
        raise ValueError(
            f"Model result missing 'output' key: {parsed!r}\n"
            f"--- subprocess stdout ---\n{stdout}\n"
            f"--- subprocess stderr ---\n{stderr}"
        )
    return parsed["output"]


class ModelProxy:
    def __init__(
        self,
        project,
        env_name,
        model=None,
        runtime="conda",
        timeout=DEFAULT_TIMEOUT,
        load_kwargs=None,
    ):
        self.project = project
        self.model = model
        self.runtime = runtime
        self.env_name = env_name
        self.timeout = timeout
        self.load_kwargs = load_kwargs or {}
        # Selecting the backend here fails fast on an unsupported runtime.
        self.backend = get_backend(runtime, env_name)
        if runtime == "apptainer":
            self._process_manager = ApptainerProcessManager(
                backend=self.backend,
                project=project,
                model=model,
                load_kwargs=self.load_kwargs,
                timeout=timeout,
                run_subprocess=lambda *args, **kwargs: _run_subprocess(*args, **kwargs),
                extract_frame=lambda *args, **kwargs: _extract_frame(*args, **kwargs),
                resolve_resources=lambda: _resolve_resources(project, model),
            )
        else:
            self._process_manager = CondaProcessManager(
                backend=self.backend,
                project=project,
                model=model,
                load_kwargs=self.load_kwargs,
                timeout=timeout,
                resolve_resources=lambda: _resolve_resources(project, model),
            )

    def _ensure_backend_available(self):
        if not self._process_manager.is_started:
            if self.runtime == "conda" and not self.backend.is_runtime_available():
                raise RuntimeError(
                    "The Conda executable is not available. Install Conda and "
                    "ensure 'conda' is in PATH."
                )
            if not self.backend.is_available():
                if self.runtime == "apptainer":
                    raise RuntimeError(
                        f"The Apptainer image '{self.env_name}' does not exist. "
                        f"Please run 'vh-checker create-apptainer-image "
                        f"{self.project}' first."
                    )
                raise RuntimeError(
                    f"The environment '{self.env_name}' does not exist. "
                    f"Please run 'vh-checker create-env {self.project}' first."
                )
            if self.runtime == "apptainer" and not self.backend.is_runtime_available():
                if getattr(self.backend, "use_lima", False):
                    raise RuntimeError(
                        "Lima is not available. On macOS, install it with "
                        "'brew install lima' and ensure 'limactl' is in PATH."
                    )
                raise RuntimeError(
                    "The Apptainer executable is not available. Install Apptainer "
                    "and ensure 'apptainer' is in PATH."
                )

    def embed(self, input, **kwargs):
        self._ensure_backend_available()
        raw_result = self._process_manager.embed(
            input=input,
            kwargs=kwargs,
            cwd=Path.cwd(),
        )
        return _unwrap_model_result(raw_result)

    def predict(self, input, embedding=None, **kwargs):
        self._ensure_backend_available()
        raw_result = self._process_manager.predict(
            input=input,
            embedding=embedding,
            kwargs=kwargs,
            cwd=Path.cwd(),
        )
        return _unwrap_model_result(raw_result)

    def close(self):
        """Release this proxy's persistent worker."""
        self._process_manager.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def load_model(
    project,
    model=None,
    runtime="conda",
    image_path=None,
    ray_options=None,
    **load_kwargs,
):
    registry = _get_registry()
    if not registry.has_model(project):
        raise ValueError(f"Model '{project}' not found.")

    manifest = registry.get_model(project)
    conda_runtime = manifest.runtimes.conda
    env_name = conda_runtime.env_name if conda_runtime else f"vhmodels-{project}"
    if runtime == "apptainer":
        # Match the default output of ``create-apptainer-image``. Resolve the
        # path now so changing the working directory between load and embed
        # cannot silently select a different image.
        image_path = image_path or f"{env_name}.sif"
        env_name = str(Path(image_path).expanduser().resolve())
    elif image_path is not None and runtime != "ray":
        raise ValueError(
            "image_path can only be used with runtime='apptainer' or runtime='ray'."
        )

    if runtime == "ray":
        # Ray is an optional extra, so it is only imported for this runtime.
        from vhmodels.vh_ray.client import RayModel
        from vhmodels.vh_ray.envs import check_model_venv

        if image_path is None:
            check_model_venv(project, manifest)
        else:
            image_path = Path(image_path).expanduser().resolve()
            if not image_path.is_file():
                raise RuntimeError(f"Apptainer image '{image_path}' does not exist.")
        variant, resources = _resolve_resources(project, model)
        return RayModel(
            project=project,
            model=variant,
            resources=resources,
            load_kwargs=load_kwargs,
            ray_options=ray_options,
            image_path=image_path,
        )
    if ray_options is not None:
        raise ValueError("ray_options can only be used with runtime='ray'.")

    return ModelProxy(
        project=project,
        env_name=env_name,
        model=model,
        runtime=runtime,
        load_kwargs=load_kwargs,
    )
