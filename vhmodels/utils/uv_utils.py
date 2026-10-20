"""uv dependency spec and cache shared by Apptainer image builds and Ray model venvs."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def uv_build_cache_dir():
    """Return the host uv cache used for image builds and model venvs."""
    configured = os.environ.get("UV_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return (cache_root / "vhmodels" / "apptainer" / "uv").resolve()


# Python and Ray of every model environment (Apptainer images and Ray venvs).
# Ray workers must match the driver exactly, so these must match the
# repository's .python-version and the ray extra in pyproject.toml.
MODEL_PYTHON = "3.11.15"
RAY_VERSION = "2.59.0"


@dataclass(frozen=True)
class UvSpec:
    """A model's uv dependency spec; every field is None when it has none."""

    requirements_path: Optional[Path] = None
    exclude_path: Optional[Path] = None
    torch_backend: Optional[str] = None


def model_uv_spec(manifest):
    """Read the spec under ``runtimes.uv``; paths are resolved, not checked."""
    config = manifest.runtimes.uv
    if config is None:
        return UvSpec()

    model_dir = Path(manifest.model_dir)
    return UvSpec(
        requirements_path=model_dir / config.requirements,
        exclude_path=model_dir / config.exclude if config.exclude else None,
        torch_backend=config.torch_backend,
    )
