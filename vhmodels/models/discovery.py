"""Dependency-free discovery of model directories and their class_path.

``vhmodels.vh_checker.base.BaseModel.get_class`` runs inside every isolated
model worker -- including dependency-free test fixtures with no third-party
packages installed at all (see ``tests/fixtures/persistent_worker``). This
module therefore only uses the standard library; the Pydantic-validated
manifest schema lives in :mod:`vhmodels.models.schema` and is used by
:mod:`vhmodels.models.registry` instead, which nothing in the worker path
imports.
"""

import json
import sys
from pathlib import Path

MODEL_JSON_FILENAME = "model.json"


def models_dir():
    """The ``vhmodels/models`` directory this module lives in."""
    return Path(__file__).resolve().parent


def discover(root=None):
    """Return ``{project_id: (directory, raw model.json dict)}``, unvalidated."""
    root = Path(root) if root is not None else models_dir()
    discovered = {}
    if not root.is_dir():
        return discovered

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / MODEL_JSON_FILENAME
        if not candidate.is_file():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as error:
            print(f"Warning: failed to parse {candidate}: {error}", file=sys.stderr)
            continue
        project_id = str(data.get("model", {}).get("id", entry.name)).lower()
        discovered[project_id] = (entry, data)
    return discovered


def find_class_path(project, root=None):
    """Return the registered ``implementation.class_path`` for ``project``."""
    discovered = discover(root)
    if project not in discovered:
        raise KeyError(f"Model '{project}' is not registered.")
    _, data = discovered[project]
    try:
        return data["implementation"]["class_path"]
    except KeyError:
        raise KeyError(
            f"Model '{project}' has no 'implementation.class_path' in its manifest."
        ) from None
