"""Pydantic schema for model manifests.

A model directory under ``vhmodels/models/<Model>`` is described by two kinds
of on-disk JSON documents, validated by the classes below:

- ``model.json``    -> :class:`ModelManifest`   (identity, implementation,
  runtimes, default sources -- shared by every variant)
- ``manifests/<variant>.json`` -> :class:`VariantManifest` (per-variant
  overrides, usually just the variant id)

:class:`vhmodels.models.registry.Registry` merges the two into one
:class:`ResolvedManifest` per ``(project, variant)`` pair. ``sources`` is a
dict of named, independently-typed resources (``huggingface``, ``torch_hub``,
``url``, ``git``, ``local``, ``python_package``); string fields inside a
source may contain a ``{variant}`` placeholder, substituted by the registry
during resolution.

This module is only imported host-side (CLI, ``Registry``) and by each real
model's own ``model.py``. ``vhmodels.vh_checker.base`` and ``worker`` -- which
run inside every isolated model process, including dependency-free test
fixtures -- must not import it; they use the stdlib-only
:mod:`vhmodels.models.discovery` instead.
"""

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _Strict(BaseModel):
    """Base for every manifest document: reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


# --- Sources ----------------------------------------------------------------


class HuggingFaceSource(_Strict):
    type: Literal["huggingface"] = "huggingface"
    repo_id: str
    revision: Optional[str] = None
    files: Dict[str, str] = Field(default_factory=dict)


class TorchHubSource(_Strict):
    type: Literal["torch_hub"] = "torch_hub"
    repo: str
    revision: Optional[str] = None
    entrypoint: Optional[str] = None


class URLSource(_Strict):
    type: Literal["url"] = "url"
    url: str
    sha256: Optional[str] = None
    filename: Optional[str] = None


class GitSource(_Strict):
    type: Literal["git"] = "git"
    url: str
    revision: Optional[str] = None


class LocalSource(_Strict):
    type: Literal["local"] = "local"
    path: str


class PythonPackageSource(_Strict):
    type: Literal["python_package"] = "python_package"
    name: str
    version: Optional[str] = None


Source = Annotated[
    Union[
        HuggingFaceSource,
        TorchHubSource,
        URLSource,
        GitSource,
        LocalSource,
        PythonPackageSource,
    ],
    Field(discriminator="type"),
]


# --- Runtimes -----------------------------------------------------------------


class CondaPlatform(_Strict):
    environment: str


class CondaRuntime(_Strict):
    env_name: str
    platforms: Dict[str, CondaPlatform]


class ApptainerRuntime(_Strict):
    platform: str
    python: str
    requirements: str
    exclude: Optional[str] = None
    torch_backend: Optional[str] = None


class Runtimes(_Strict):
    conda: Optional[CondaRuntime] = None
    apptainer: Optional[ApptainerRuntime] = None


# --- Identity / implementation -------------------------------------------------


class ModelIdentity(_Strict):
    id: str
    version: str
    description: str
    homepage: Optional[str] = None


class Implementation(_Strict):
    class_path: str
    capabilities: List[str] = Field(default_factory=lambda: ["embed"])


# --- model.json -----------------------------------------------------------------


class ModelManifest(_Strict):
    schema_version: str
    model: ModelIdentity
    implementation: Implementation
    runtimes: Runtimes
    supported_platforms: List[str]
    sources: Dict[str, Source] = Field(default_factory=dict)
    # Populated by Registry after loading; absent from the on-disk JSON.
    model_dir: str = ""


# --- manifests/<variant>.json -----------------------------------------------------


class VariantManifest(_Strict):
    variant: str
    description: Optional[str] = None
    sources: Dict[str, Source] = Field(default_factory=dict)


# --- fully resolved, ready-to-run manifest ----------------------------------------


class ResolvedManifest(_Strict):
    schema_version: str
    model: ModelIdentity
    implementation: Implementation
    runtimes: Runtimes
    supported_platforms: List[str]
    variant: str
    description: str
    sources: Dict[str, Source]
    model_dir: str
