---
name: model-search
description: Discovers and selects Helmholtz Munich VH CHC embedding models (DinoBloom microscopy, Hyformer molecules/peptides, ProtTrans proteins, MolE SMILES). Use when choosing a model, listing models, comparing modalities, or before embed. Triggers on vhmodels, vh-checker, virtual-human-chc, DinoBloom, Hyformer, ProtTrans, MolE, model zoo, or which model should I use.
---

# VH model discovery

`vhmodels` not on PyPI. Resolve package before `vh-checker list`.

**Platforms:** Linux and macOS only. No Windows support.

## Package location

1. `vh-checker list` or `python -c "import vhmodels"` — OK → stop; do not filesystem-search.
2. Else ask user for repo root, or find `pyproject.toml` with `name = "vhmodels"`.
3. `pip install -e "<repo_root>.[cli]"` (or `uv pip install -e "<repo_root>.[cli]"`)
4. Source: https://github.com/HelmholtzAI-Consultants-Munich/vhmodels/

## List models

```bash
vh-checker list
```

Authoritative registry. Do not hardcode checkpoint names — they change. Optional checkout path: `vhmodels/models/*/model.json` (+ `manifests/*.json` for per-variant sources).

## Modality → project

| Data | Project | Notes |
|------|---------|-------|
| Cell / microscopy images | `dinobloom` | [HF](https://huggingface.co/virtual-human-chc/DinoBloom) |
| SMILES / molecules | `hyformer` or `mole` | Hyformer: transformer checkpoints. MolE: GIN from `.smiles` file |
| Peptides | `hyformer` | peptide checkpoints |
| Protein sequences | `prottrans` | [HF collection](https://huggingface.co/collections/virtual-human-chc/prottrans) |

**Hyformer vs MolE:** Hyformer — broader peptide/molecule tasks. MolE — SMILES-file GIN embeddings. Ask if unclear.

## Out of scope

Embed, Conda/Apptainer setup → **model-embed**. `predict` / `generate` not in v0.1.

After project chosen → **model-embed** for `create-env` and `embed`.
