---
name: model-search
description: Discovers and selects Helmholtz Munich VH CHC embedding models (DinoBloom microscopy, Hyformer molecules/peptides, ProtTrans proteins, MolE SMILES). Use when choosing a model, listing models, comparing modalities, or before embed. Triggers on vhmodels, vh-checker, virtual-human-chc, DinoBloom, Hyformer, ProtTrans, MolE, model zoo, or which model should I use.
---

# VH model discovery

`vhmodels` not on PyPI. Resolve package before `vh-checker list`.

## Package location

1. `vh-checker list` or `python -c "import vhmodels"` — OK → stop; do not filesystem-search.
2. Else ask user for repo root, or find `pyproject.toml` with `name = "vhmodels"`.
3. `pip install -e /path/to/virtual_human_chc` (or `uv pip install -e …`)
4. Source: https://github.com/HelmholtzAI-Consultants-Munich/virtual_human_chc/tree/main

## List models

```bash
vh-checker list
```

Registry: `vhmodels/models/*/config.json` in checkout.

## Modality → project

| Data | Project | Notes |
|------|---------|-------|
| Cell / microscopy images | `dinobloom` | Variants `s` `b` `l` `g`. [HF](https://huggingface.co/virtual-human-chc/DinoBloom) |
| SMILES / molecules | `hyformer` or `mole` | Hyformer: transformer checkpoints. MolE: GIN from `.smiles` file |
| Peptides | `hyformer` | `hyformer_peptides_34M`, `hyformer_peptides_34_MIC` |
| Protein sequences | `prottrans` | [HF collection](https://huggingface.co/collections/virtual-human-chc/prottrans) |

**Hyformer vs MolE:** Hyformer — broader peptide/molecule tasks. MolE — SMILES-file GIN embeddings. Ask if unclear.

**Hyformer checkpoints:** `hyformer_molecules_50M`, `hyformer_molecules_8M`, `hyformer_peptides_34M`, `hyformer_peptides_34_MIC`

**ProtTrans (common):** `prot_t5_xl_uniref50`, `prot_t5_xxl_uniref50`, `prot_bert`, `prot_albert`, `prot_xlnet`, `prot_electra_bfd` — full list in `vhmodels/models/ProtTrans/config.json`.

## Out of scope

Embed, Conda/Docker setup → **model-embed**. `predict` / `generate` not in v0.1.

After project chosen → **model-embed** for `create-env` and `embed`.
