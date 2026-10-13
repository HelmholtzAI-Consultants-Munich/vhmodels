---
name: model-search
description: Discovers and selects Helmholtz Munich VH CHC embedding models (DinoBloom microscopy, H-Optimus-0 histology, Hyformer molecules/peptides, ProtTrans proteins, MolE antimicrobial activity, Nicheformer spatial transcriptomics). Use when choosing a model, listing models, comparing modalities, or before embed. Triggers on vhmodels, vh-checker, virtual-human-chc, DinoBloom, H-Optimus-0, Hyformer, ProtTrans, MolE, Nicheformer, model zoo, or which model should I use.
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
| H&E histology tiles (224x224, 0.5 µm/px) | `hoptimus0` | Gated [HF model](https://huggingface.co/bioptimus/H-optimus-0); needs HF login |
| SMILES / molecules | `hyformer` or `mole` | Hyformer: transformer checkpoints. MolE: GIN from `chem_name`/`smiles` TSV, plus antimicrobial `predict` |
| Peptides | `hyformer` | peptide checkpoints |
| Protein sequences | `prottrans` | [HF collection](https://huggingface.co/collections/virtual-human-chc/prottrans) |
| Spatial transcriptomics (`.h5ad`) | `nicheformer` | Needs a technology-mean `.npy` |

**Hyformer vs MolE:** Hyformer — broader peptide/molecule tasks. MolE — GIN embeddings and antimicrobial activity prediction. Ask if unclear.

## Out of scope

Embed, Conda/Apptainer setup → **model-embed**. `predict` only for MolE; `generate` not in v0.1.

After project chosen → **model-embed** for `create-env` and `embed`.
