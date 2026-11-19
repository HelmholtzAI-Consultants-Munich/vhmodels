"""Builds the human-readable model catalog, ``vhmodels/models/README.md``.

The catalog joins two sources:

- the registry (``model.json`` and ``manifests/<variant>.json``): input,
  capabilities, platforms and, per variant, the parameter count;
- each variant's Hugging Face model card (``card.repo_id`` in the manifest):
  license, modality, citation and developer, read from the card metadata keys
  in ``CARD_KEYS``.

``card.overrides`` in a manifest take precedence over the card. A missing field
is an error for a card in ``OWN_ORG`` and a warning for an external card, which
only ``card.overrides`` can complete.

Only ``vh-checker catalog`` uses this module. Reading the cards needs network
access to the Hugging Face Hub; everything else works offline.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

OWN_ORG = "virtual-human-chc"

# Catalog field -> Hugging Face card metadata key.
CARD_KEYS = {
    "license": "license",
    "modality": "vh_modality",
    "citation": "vh_citation",
    "developer": "vh_developer",
}


@dataclass
class Problem:
    level: str  # "error" or "warning"
    message: str


@dataclass
class Variant:
    id: str
    params: Optional[str] = None
    repo_id: Optional[str] = None
    gated: bool = False
    fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class Model:
    project: str
    name: str
    manifest: object  # ModelManifest
    variants: List[Variant]

    def values(self, name):
        """Distinct values of one card field across the variants, in order."""
        values = []
        for variant in self.variants:
            value = variant.fields.get(name)
            if value and value not in values:
                values.append(value)
        return values


# --- Hugging Face ----------------------------------------------------------------


def fetch_card(repo_id):
    """Return the ``CARD_KEYS`` fields and the gated flag of one model card."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id)
    metadata = info.card_data.to_dict() if info.card_data else {}
    card = {name: metadata.get(key) for name, key in CARD_KEYS.items()}
    card["gated"] = bool(info.gated)
    return card


def list_org_repos(org=OWN_ORG):
    """Return the ids of all model repos of a Hugging Face organization."""
    from huggingface_hub import HfApi

    return [model.id for model in HfApi().list_models(author=org)]


# --- Joining the registry with the cards -----------------------------------------


def _text(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or None
    return str(value).strip() or None


def collect(registry, fetch=fetch_card):
    """Join every registered variant with its model card.

    ``fetch(repo_id)`` returns a dict with the ``CARD_KEYS`` fields and
    ``gated``. Returns ``(models, problems)``.
    """
    cards = {}
    reported = set()
    models = []
    problems = []
    for project, manifest in sorted(registry.list_models().items()):
        variants = []
        for variant_id in registry.list_variants(project):
            resolved = registry.resolve(project, variant_id)
            card = resolved.card
            if card is None:
                problems.append(
                    Problem("error", f"{project}/{variant_id}: no card in the manifest")
                )
                variants.append(Variant(variant_id, resolved.params))
                continue

            if card.repo_id not in cards:
                try:
                    cards[card.repo_id] = fetch(card.repo_id)
                except Exception as error:
                    cards[card.repo_id] = None
                    problems.append(
                        Problem(
                            "error", f"{card.repo_id}: cannot read the card ({error})"
                        )
                    )
            fetched = cards[card.repo_id] or {}

            fields = {name: _text(fetched.get(name)) for name in CARD_KEYS}
            fields.update(card.overrides.model_dump(exclude_none=True))
            fields = {name: value for name, value in fields.items() if value}

            missing = [name for name in CARD_KEYS if name not in fields]
            key = (card.repo_id, tuple(missing))
            if missing and cards[card.repo_id] is not None and key not in reported:
                reported.add(key)
                if card.repo_id.split("/")[0] == OWN_ORG:
                    keys = ", ".join(CARD_KEYS[name] for name in missing)
                    problems.append(
                        Problem("error", f"{card.repo_id}: card metadata lacks {keys}")
                    )
                else:
                    problems.append(
                        Problem(
                            "warning",
                            f"{card.repo_id}: no {', '.join(missing)}; add them to "
                            f"card.overrides in {project}'s model.json",
                        )
                    )

            variants.append(
                Variant(
                    variant_id,
                    resolved.params,
                    card.repo_id,
                    bool(fetched.get("gated")),
                    fields,
                )
            )
        models.append(Model(project, Path(manifest.model_dir).name, manifest, variants))
    return models, problems


def unused_repos(registry, org_repos):
    """Warn about repos in ``org_repos`` that no manifest refers to."""
    used = set()
    for project in registry.list_models():
        for variant in registry.list_variants(project):
            resolved = registry.resolve(project, variant)
            if resolved.card is not None:
                used.add(resolved.card.repo_id)
            for source in resolved.sources.values():
                if source.type == "huggingface":
                    used.add(source.repo_id)
    return [
        Problem("warning", f"{repo}: on Hugging Face but in no manifest")
        for repo in sorted(set(org_repos) - used)
    ]


# --- Markdown ----------------------------------------------------------------------

HEADER = """\
# Model catalog

<!-- Generated by `vh-checker catalog` from the model manifests and their
Hugging Face model cards. Do not edit by hand: change the manifest or the card
and run the command again. -->

"""


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _row(cells):
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _anchor(heading):
    return re.sub(r"[^\w-]", "", heading.strip().lower().replace(" ", "-"))


def _sentence(text):
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _link(url):
    return f"<{url}>" if url.startswith(("http://", "https://")) else url


def _input(manifest):
    spec = manifest.implementation.input
    if spec is None:
        return "—"
    if spec.keys:
        return f"{spec.type} ({', '.join(spec.keys)})"
    return spec.type


def _load_call(model):
    first = model.variants[0].id if model.variants else None
    if len(model.variants) == 1:
        return f'vhmodels.load_model(project="{model.project}")'
    return f'vhmodels.load_model(project="{model.project}", model="{first}")'


def render(models):
    """Return the catalog Markdown for the models returned by :func:`collect`."""
    lines = [HEADER.rstrip("\n"), ""]
    lines.append(
        _row(["Model", "Modality", "Input", "Variants", "License", "Developer"])
    )
    lines.append(_row(["---"] * 6))
    for model in models:
        lines.append(
            _row(
                [
                    f"[{model.name}](#{_anchor(model.name)})",
                    ", ".join(model.values("modality")) or "—",
                    _input(model.manifest),
                    len(model.variants),
                    ", ".join(model.values("license")) or "—",
                    ", ".join(model.values("developer")) or "—",
                ]
            )
        )

    for model in models:
        manifest = model.manifest
        lines += ["", f"## {model.name}", "", _sentence(manifest.model.description), ""]
        lines.append(f"- **Project:** `{model.project}`")
        lines.append(
            "- **Capabilities:** "
            + ", ".join(f"`{name}`" for name in manifest.implementation.capabilities)
        )
        lines.append(f"- **Input:** {_input(manifest)}")
        lines.append(f"- **Platforms:** {', '.join(manifest.supported_platforms)}")
        citations = model.values("citation")
        lines.append(
            "- **Citation:** " + (", ".join(_link(url) for url in citations) or "—")
        )
        if any(variant.gated for variant in model.variants):
            lines.append(
                "- **Access:** gated on Hugging Face; request access on the model "
                "card and log in with `hf auth login`"
            )
        lines += ["", _row(["Variant", "Parameters", "Model card"]), _row(["---"] * 3)]
        for variant in model.variants:
            card = (
                f"[{variant.repo_id}](https://huggingface.co/{variant.repo_id})"
                if variant.repo_id
                else "—"
            )
            lines.append(_row([f"`{variant.id}`", variant.params or "—", card]))
        lines += [
            "",
            "```python",
            "import vhmodels",
            "",
            f"model = {_load_call(model)}",
            "```",
        ]
    return "\n".join(lines) + "\n"
