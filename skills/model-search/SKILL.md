---
name: model-search
description: Discover and choose a Helmholtz Munich Virtual Human CHC deep-learning model for embeddings (DinoBloom cell microscopy, Hyformer molecules/peptides, ProtTrans proteins, MolE SMILES). Use when the user needs to find, compare, or select a VH model, asks what models exist, wants model metadata or HuggingFace links, or is unsure which model fits their modality. Trigger on vhmodels, vh-checker, virtual-human-chc, DinoBloom, Hyformer, ProtTrans, MolE, model zoo, protein/cell/molecule embeddings, or "which model should I use".
---

# VH model discovery

`vhmodels` is **not on PyPI yet** (early prototype). Resolve install location before `vh-checker list` (see **Package location** below).

## Package location

Run once per session before any `vh-checker` / `vhmodels` command:

1. **Already works?** `vh-checker list` or `python -c "import vhmodels"` — if OK, **skip install**; do not search the filesystem.
2. Else **ask the user** for the repo root, or search cwd/parent dirs for `pyproject.toml` with `name = "vhmodels"`.
3. **Install editable** from that root:

```bash
pip install -e /path/to/virtual_human_chc
# or: uv pip install -e /path/to/virtual_human_chc
```

Verify: `vh-checker list`

**Source (if user asks where to get it):** https://github.com/HelmholtzAI-Consultants-Munich/virtual_human_chc/tree/main

Clone when needed:

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/virtual_human_chc.git
cd virtual_human_chc && pip install -e .
```

## List registered models

```bash
vh-checker list
```

Registry is also defined under `vhmodels/models/*/config.json` in the checkout.

## Modality → project

| User has… | Project | Notes |
|-----------|---------|-------|
| Cell / microscopy images (bmp, png, jpg) | `dinobloom` | Variants `s`, `b`, `l`, `g` (size). HF: [DinoBloom](https://huggingface.co/virtual-human-chc/DinoBloom) |
| SMILES / small molecules (strings or file) | `hyformer` or `mole` | Hyformer: multiple checkpoints; generative+predictive transformer. MolE: GIN molecular representations. |
| Peptide sequences | `hyformer` | e.g. `hyformer_peptides_34M`, `hyformer_peptides_34_MIC` |
| Protein amino-acid sequences | `prottrans` | Many checkpoints (T5, BERT, ALBERT, XLNet, ELECTRA). HF collection: [ProtTrans](https://huggingface.co/collections/virtual-human-chc/prottrans) |

**Hyformer vs MolE:** Hyformer — joint transformer, broader peptide/molecule tasks. MolE — task-independent GIN embeddings from SMILES files. Ask user goal if unclear.

## Hyformer checkpoints

- `hyformer_molecules_50M`, `hyformer_molecules_8M`
- `hyformer_peptides_34M`, `hyformer_peptides_34_MIC`

## ProtTrans checkpoints (common)

- `prot_t5_xl_uniref50`, `prot_t5_xxl_uniref50`, `prot_t5_xl_bfd`
- `prot_bert`, `prot_bert_bfd`, `prot_albert`, `prot_xlnet`
- `prot_electra_bfd`, `prot_electra_generator_bfd`, `prot_electra_discriminator_bfd`

Full list: `config.json` in `vhmodels/models/ProtTrans/` or HuggingFace `virtual-human-chc/prot_*` repos.

## What this skill does *not* cover

- Running embeddings, Conda/Docker env setup → **model-embed**
- `predict` / `generate` — not implemented yet in v0.1

## Optional next step

After the user picks a project → load **model-embed** to create the runtime env and run `embed`.
