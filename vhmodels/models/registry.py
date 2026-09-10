"""Discovers, validates, and resolves model manifests.

``Registry`` combines each model's ``model.json`` with one of its
``manifests/<variant>.json`` files into a single, fully validated
:class:`~vhmodels.models.schema.ResolvedManifest`. String fields inside a
resolved source may reference ``{variant}``, substituted with the requested
variant id (see ``ProtTrans/model.json``'s ``"repo_id": "virtual-human-chc/
{variant}"`` for the canonical example).

This module requires Pydantic and is only imported host-side (CLI, the
subprocess launcher in ``vh_checker.factory``) and by each real model's own
``model.py``, executing inside its already dependency-heavy environment. It
is never imported by ``vh_checker.base`` or ``vh_checker.worker``, which must
keep working in dependency-free workers -- see :mod:`vhmodels.models.discovery`.
"""

import json
import sys
from pathlib import Path
from typing import Dict

from pydantic import TypeAdapter, ValidationError

from vhmodels.models import discovery
from vhmodels.models.schema import (
    ModelManifest,
    ResolvedManifest,
    Source,
    VariantManifest,
)

_SOURCE_ADAPTER = TypeAdapter(Source)


def _substitute(value, variant):
    """Recursively replace ``{variant}`` in every string leaf."""
    if isinstance(value, str):
        return value.replace("{variant}", variant)
    if isinstance(value, dict):
        return {key: _substitute(item, variant) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, variant) for item in value]
    return value


class Registry:
    """Discovers ``vhmodels/models/*/model.json`` and resolves variants."""

    def __init__(self, models_dir=None):
        self._models_dir = (
            Path(models_dir) if models_dir is not None else discovery.models_dir()
        )
        self._models: Dict[str, ModelManifest] = {}
        for project_id, (directory, raw) in discovery.discover(
            self._models_dir
        ).items():
            try:
                manifest = ModelManifest.model_validate(raw)
            except ValidationError as error:
                print(
                    f"Warning: invalid manifest for '{project_id}' at "
                    f"{directory / discovery.MODEL_JSON_FILENAME}: {error}",
                    file=sys.stderr,
                )
                continue
            self._models[project_id] = manifest.model_copy(
                update={"model_dir": str(directory)}
            )

    def list_models(self):
        """Return ``{project_id: ModelManifest}`` for every valid model."""
        return dict(self._models)

    def has_model(self, project):
        return project in self._models

    def get_model(self, project):
        """Return the model-level manifest for ``project``."""
        try:
            return self._models[project]
        except KeyError:
            raise ValueError(f"Model '{project}' is not registered.") from None

    def list_variants(self, project):
        """Return the sorted variant ids declared under ``manifests/``."""
        manifest = self.get_model(project)
        manifests_dir = Path(manifest.model_dir) / "manifests"
        if not manifests_dir.is_dir():
            return []
        return sorted(path.stem for path in manifests_dir.glob("*.json"))

    def resolve(self, project, variant=None):
        """Merge the model-level manifest with one variant into one manifest."""
        manifest = self.get_model(project)
        variants = self.list_variants(project)
        if not variants:
            raise ValueError(
                f"Model '{project}' has no variant manifests under 'manifests/'."
            )
        if variant is None:
            if len(variants) > 1:
                raise ValueError(
                    f"Model '{project}' requires a variant. Available: {variants}"
                )
            variant = variants[0]
        elif variant not in variants:
            raise ValueError(
                f"Unknown variant '{variant}' for model '{project}'. "
                f"Available: {variants}"
            )

        variant_manifest = self._load_variant(manifest, variant)

        merged_sources = dict(manifest.sources)
        merged_sources.update(variant_manifest.sources)
        resolved_sources = {
            name: _SOURCE_ADAPTER.validate_python(
                _substitute(source.model_dump(), variant)
            )
            for name, source in merged_sources.items()
        }

        return ResolvedManifest(
            schema_version=manifest.schema_version,
            model=manifest.model,
            implementation=manifest.implementation,
            runtimes=manifest.runtimes,
            supported_platforms=manifest.supported_platforms,
            variant=variant,
            description=variant_manifest.description or manifest.model.description,
            sources=resolved_sources,
            model_dir=manifest.model_dir,
        )

    @staticmethod
    def _load_variant(manifest, variant):
        path = Path(manifest.model_dir) / "manifests" / f"{variant}.json"
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return VariantManifest.model_validate(raw)


# Eagerly discovered singleton, mirroring the previous module-level
# ``MODEL_REGISTRY``. Host-side code (CLI, factory) shares this instance;
# tests construct their own ``Registry(models_dir=...)`` to isolate fixtures.
REGISTRY = Registry()
