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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    revision: str = Field(min_length=1)


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


class UvRuntime(_Strict):
    """uv dependency spec shared by the Apptainer image and the Ray venv.

    The Python version is not per model: see ``utils.uv_utils.MODEL_PYTHON``.
    """

    requirements: str
    exclude: Optional[str] = None
    torch_backend: Optional[str] = None


class ApptainerRuntime(_Strict):
    platform: str


class Runtimes(_Strict):
    conda: Optional[CondaRuntime] = None
    uv: Optional[UvRuntime] = None
    apptainer: Optional[ApptainerRuntime] = None

    @model_validator(mode="after")
    def _apptainer_needs_uv(self):
        if self.apptainer is not None and self.uv is None:
            raise ValueError("runtimes.apptainer requires runtimes.uv")
        return self


# --- Identity / implementation -------------------------------------------------


class ModelIdentity(_Strict):
    id: str
    version: str
    description: str
    homepage: Optional[str] = None


class ModelInput(_Strict):
    """Shape of the ``input`` that ``embed`` and ``predict`` take.

    - ``sequences``: list of strings (SMILES, amino-acid sequences)
    - ``file``: one path
    - ``images``: an image path, a folder path or a list of image paths
    - ``named_files``: dict of paths with exactly the names in ``keys``
    """

    type: Literal["sequences", "file", "images", "named_files"]
    keys: Optional[List[str]] = None

    @model_validator(mode="after")
    def _keys_only_for_named_files(self):
        if (self.type == "named_files") != bool(self.keys):
            raise ValueError("input.keys is required for, and only for, named_files")
        return self

    def check(self, input):
        """Raise ``ValueError`` if ``input`` does not have this shape."""
        is_strings = isinstance(input, list) and all(isinstance(i, str) for i in input)
        if self.type == "sequences":
            valid, expected = is_strings, "a list of strings"
        elif self.type == "file":
            valid, expected = isinstance(input, str), "a file path"
        elif self.type == "images":
            valid = isinstance(input, str) or is_strings
            expected = "an image or folder path, or a list of image paths"
        else:
            valid = (
                isinstance(input, dict)
                and set(input) == set(self.keys)
                and all(isinstance(path, str) for path in input.values())
            )
            expected = f"a dict of paths with keys {self.keys}"
        if not valid:
            raise ValueError(f"Expected input: {expected}; got {input!r:.100}.")


class Implementation(_Strict):
    class_path: str
    capabilities: List[str] = Field(default_factory=lambda: ["embed"])
    # Optional so that manifests without it skip the input check.
    input: Optional[ModelInput] = None


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
