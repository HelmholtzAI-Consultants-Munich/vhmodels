---
name: model-embed
description: Runs VH CHC model embeddings (DinoBloom, Hyformer, ProtTrans, MolE) via vhmodels load_model/embed and isolated Conda or Apptainer runtimes. Use when embedding images, proteins, peptides, or molecules; calling vh-checker create-env; or after model-search selected a project. Triggers on embed, load_model, create-env, vh-checker, vhmodels runtime setup.
---

# VH model embed

`vhmodels` = thin host proxy + **per-model Conda/Apptainer runtime** (torch, RDKit, CUDA). Not on PyPI — resolve package first (same steps as **model-search**).

**Platforms:** Linux and macOS only. No Windows support.

## Agent rules

- **API only:** `vhmodels.load_model`, `vh-checker` — never import `vhmodels.models.*`.
- **Before embed:** `conda --version` or `apptainer --version` must work; then `vh-checker create-env <project>` (conda) or `vh-checker create-apptainer-image <project>` (Apptainer).
- **Miniforge / conda-forge** — not Anaconda Miniconda (org licensing blocks).
- v0.1: **`embed` only**; `predict` / `generate` are stubs.
- First run: large downloads + GPU optional — warn user.
- **Docker not supported.** Use Conda (default) or Apptainer.

## Workflow

```
package OK  →  conda OR apptainer  →  create-env | create-apptainer-image  →  load_model  →  embed
```

1. **Package** — `vh-checker list` works? Stop. Else `pip install -e "<repo_root>.[cli]"`. Source: https://github.com/HelmholtzAI-Consultants-Munich/vhmodels/
2. **Runtime** — missing? Read [environment-setup.md](references/environment-setup.md), run `scripts/install-miniforge.sh` (user consent), `source ~/miniforge3/etc/profile.d/conda.sh`.
3. **Model env** — `vh-checker create-env <project>` (conda) or `create-apptainer-image <project>` (Apptainer).
4. **Embed** — examples below.

Unsure which project? Load **model-search** first.

## Embed

```python
import vhmodels

model = vhmodels.load_model(project="dinobloom", model="s")
model.embed(input="path/to/image.bmp")

model = vhmodels.load_model(project="hyformer", model="hyformer_molecules_50M")
model.embed(input=["CCO", "c1ccccc1"])  # list of SMILES, not a file path

model = vhmodels.load_model(project="prottrans", model="prot_t5_xl_uniref50")
model.embed(input=["MKVILLLLAVVAFGHALCRV", "PRTEINO"])

model = vhmodels.load_model(project="mole")
model.embed(input="example_data/MolE/sequences.smiles")  # path to .smiles file

model = vhmodels.load_model(project="mole", runtime="singularity")
```

Returns a list from subprocess JSON `output`. Example data: repo `example_data/`.

## Input types

| Project | `embed(input=…)` |
|---------|------------------|
| `dinobloom` | Image path, image directory, or list of paths |
| `hyformer` | **List of SMILES strings** |
| `prottrans` | **List of protein sequences** |
| `mole` | **Path to `.smiles` file** |

## Scripts

**`scripts/install-miniforge.sh`** — execute when conda missing and user agrees. Picks OS/arch Miniforge build, installs to `~/miniforge3`. Details: [environment-setup.md](references/environment-setup.md).

## Troubleshooting

| Issue | Action |
|-------|--------|
| No conda/apptainer | [environment-setup.md](references/environment-setup.md) + install script |
| Env does not exist | `vh-checker create-env <project>` |
| SIF missing | `vh-checker create-apptainer-image <project>` |
| Host import errors | Normal — runs in model sub-env |

Optional: embeddings → KG pipeline → suggest **biotope** after user confirms.
