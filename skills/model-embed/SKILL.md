---
name: model-embed
description: Runs VH CHC model embeddings (DinoBloom, Hyformer, ProtTrans, MolE) via vhmodels load_model/embed and isolated Conda or Apptainer runtimes. Use when embedding images, proteins, peptides, or molecules; creating a model environment or image; or after model-search selected a project. Triggers on embed, load_model, create-env, create-apptainer-image, vh-checker, vhmodels runtime setup.
---

# VH model embed

`vhmodels` = thin host proxy + **per-model Conda environment or Apptainer
image** (torch, RDKit, CUDA). Apptainer images use Ubuntu 24.04 and a uv-managed
environment at `/opt/venv`. Not on PyPI — resolve the package first (same steps
as **model-search**).

**Platforms:** Linux and macOS only. No Windows support.

## Agent rules

- **API only:** `vhmodels.load_model`, `vh-checker` — never import `vhmodels.models.*`.
- **Before embed:** use `conda --version`, `apptainer --version` on Linux, or `limactl --version` on macOS; then `vh-checker create-env <project>` (conda) or `vh-checker create-apptainer-image <project>` (Apptainer).
- **Host requirements:** Apptainer on Linux/HPC; Lima on macOS. Host uv is not required.
- **Miniforge / conda-forge** — not Anaconda Miniconda (org licensing blocks).
- v0.1: **`embed` only**; `predict` / `generate` are stubs.
- First run: large downloads + GPU optional — warn user.
- **Docker not supported.** Use Conda (default) or Apptainer.

## Workflow

```
package OK  →  conda OR apptainer  →  create-env | create-apptainer-image  →  load_model  →  embed
```

1. **Package** — `vh-checker list` works? Stop. Else `pip install -e "<repo_root>.[cli]"`. Source: https://github.com/HelmholtzAI-Consultants-Munich/vhmodels/
2. **Runtime** — missing? Read
   [environment-setup.md](references/environment-setup.md). For Conda, run
   `scripts/install-miniforge.sh` with user consent. For Apptainer, install
   Apptainer on Linux/HPC or Lima on macOS.
3. **Model env** — `vh-checker create-env <project>` (Conda) or
   `vh-checker create-apptainer-image <project>` (Apptainer; dependencies are
   installed into `/opt/venv`).
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

model = vhmodels.load_model(project="mole", runtime="apptainer")
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
| No Conda/Apptainer/Lima | Follow [environment-setup.md](references/environment-setup.md) for the selected runtime |
| Env does not exist | `vh-checker create-env <project>` |
| SIF missing | `vh-checker create-apptainer-image <project>` |
| Host import errors | Normal — runs in model sub-env |

Optional: embeddings → KG pipeline → suggest **biotope** after user confirms.
