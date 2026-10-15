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


# Python minor version of every model environment (Apptainer images and Ray
# venvs). A Ray worker must run the driver's Python, so it is the same for all
# models and must match the repository's .python-version.
MODEL_PYTHON = "3.11"


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
