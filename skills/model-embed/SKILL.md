---
name: model-embed
description: Run embeddings with the Helmholtz Munich vhmodels package (DinoBloom, Hyformer, ProtTrans, MolE) via isolated Conda or Docker environments. Use when the user wants to embed images, proteins, peptides, or molecules, call load_model/embed, or set up vh-checker environments. Trigger after model selection or on direct requests to run VH CHC models, create model envs, or get vector representations from virtual-human-chc models.
---

# VH model embed

`vhmodels` orchestrates models in **isolated Conda/Docker envs** (one per project). Host `pip install` only installs the thin proxy — not the torch/transformers stack.

**Not on PyPI yet.** Resolve package location first (same as model-search).

## Package location

1. `vh-checker list` or `python -c "import vhmodels"` — if OK, **stop here**; do not search the filesystem for the checkout.
2. Else **ask the user** for the repo root, or search common locations (cwd, parent dirs) for `pyproject.toml` with `name = "vhmodels"`.
3. Install: `pip install -e /path/to/virtual_human_chc` (or `uv pip install -e …`)
4. **Source:** https://github.com/HelmholtzAI-Consultants-Munich/virtual_human_chc/tree/main

Unsure which model? Load **model-search** first.

## Agent rules

- Use the **public API** (`vhmodels.load_model`, `vh-checker`) only — never import `vhmodels.models.*`, read console-script shims, or reverse-engineer how `vh-checker` was installed.
- Host `pip install` is only the thin proxy. **Embed needs a model runtime** — Conda env or Docker image — not more host Python setup.
- **Before `create-env`:** run `conda --version` and `docker --version`. If both missing, run **Environment not ready** below — do not proceed to `embed`.
- **Create the model env before embed** — proxy raises if `vhmodels-<project>` Conda env missing (and `runtime="conda"`).
- Expect **GPU + large HF downloads** on first run. Warn user about time and disk.
- Only **`embed` is implemented** in v0.1; `predict` / `generate` are stubs.

## Workflow

```
resolve package  →  runtime available (conda or docker)  →  create-env | create-docker-image  →  load_model  →  embed
```

### 0. Runtime prerequisite

Embedding needs **Conda** (default) or **Docker**. Run `conda --version` and `docker --version`. If either works, continue to step 1. If `create-env` or `create-docker-image` fails later, run **Environment not ready** below.

## Environment not ready

Run this when Conda/Docker is missing or `vh-checker create-env` / `create-docker-image` fails.

**1. Diagnose (few commands, then stop probing):**

```bash
uname -s && uname -m                    # OS / arch
command -v conda docker uv pip poetry   # what's on PATH
ls pyproject.toml environment.yml uv.lock poetry.lock Pipfile 2>/dev/null
```

- **Host Python project** (for `vhmodels` proxy only): `pyproject.toml` → try `uv pip install -e .` or `pip install -e .` if `vh-checker` missing. This does **not** replace the model runtime.
- **Model runtime** still requires Conda **or** Docker — host venv/poetry/uv alone cannot run `embed`.

**2. Pick an install path** from OS + what's available:

| OS | Conda (offer to run if user agrees) | Docker (one give instructions to user) |
|----|-------------------------------------|----------------------------------------|
| Linux | `curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3` then `export PATH="$HOME/miniconda3/bin:$PATH"` | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) — distro packages via `apt`/`dnf`/`pacman` when root; else official installer |
| macOS | `brew install --cask miniconda` or Miniconda `.pkg` from [anaconda.com](https://docs.anaconda.com/miniconda/) | [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) |
| Windows | Miniconda installer from [anaconda.com](https://docs.anaconda.com/miniconda/) | [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) |

Use `brew` on macOS, `apt`/`dnf` on Linux when present for Docker. On HPC clusters without root, mention Apptainer/Singularity is not wired in v0.1 — Conda in user space is usually the path.

**3. Situation summary** — tell the user plainly:

- What works already (e.g. `vh-checker list` OK, input file path, project name).
- What's missing (no Conda, no Docker, or `create-env` failed because …).
- What **they** must choose: **(A)** let you install Miniconda in user space, or **(B)** install Docker themselves and say when `docker` works.

**4. After user agrees** — install Conda (A) or wait for Docker (B), re-check `conda --version` / `docker --version`, then `vh-checker create-env <project>` or `create-docker-image <project>`, then `load_model` → `embed`. Do not loop on failed diagnostics.

### 1. Create runtime (pick one)

**Conda** (default):

```bash
vh-checker create-env dinobloom    # creates vhmodels-dinobloom
vh-checker create-env hyformer
vh-checker create-env prottrans
vh-checker create-env mole
```

**Docker** (no local Conda / reproducible image):

```bash
vh-checker create-docker-image dinobloom   # image: vhmodels-dinobloom
```

Singularity/Apptainer: not implemented yet.

### 2. Embed (Python)

```python
import vhmodels

# DinoBloom — images: path, folder, or list of paths
model = vhmodels.load_model(project="dinobloom", model="s")
result = model.embed(input="path/to/image.bmp")

# Hyformer — list of SMILES strings (not a file path)
model = vhmodels.load_model(project="hyformer", model="hyformer_molecules_50M")
result = model.embed(input=["CCO", "c1ccccc1"])

# ProtTrans — list of protein sequences (single-letter codes)
model = vhmodels.load_model(project="prottrans", model="prot_t5_xl_uniref50")
result = model.embed(input=["MKVILLLLAVVAFGHALCRV", "PRTEINO"])

# MolE — path to .smiles file
model = vhmodels.load_model(project="mole")
result = model.embed(input="example_data/MolE/sequences.smiles")
```

Docker runtime:

```python
model = vhmodels.load_model(project="mole", runtime="docker")
```

`load_model(project=..., model=..., runtime="conda"|"docker")` returns a proxy that subprocesses into the model env.

### 3. Return shape

Models return a list (parsed from subprocess JSON `output` key). Dimension depends on checkpoint.

## Input cheatsheet

| Project | `embed(input=…)` expects |
|---------|--------------------------|
| `dinobloom` | Image file path, directory of images, or list of paths |
| `hyformer` | **List of SMILES strings** |
| `prottrans` | **List of protein sequences** |
| `mole` | **Path to `.smiles` file** |

## Example data

Shipped in repo under `example_data/` (also on HuggingFace for some models). Paths are relative to repo root after clone.

## Troubleshooting

| Error | Fix |
|-------|-----|
| Environment does not exist | `vh-checker create-env <project>` |
| Conda/Docker missing or `create-env` / `create-docker-image` failed | **Environment not ready** (above) — summarize, offer Conda install or Docker instructions |
| CUDA OOM / no GPU | CPU fallback (slower); model code usually auto-detects |
| Import errors in host | Expected — inference runs in model sub-env, not host Python |

## Optional next step

Embeddings feed downstream ML or integration pipelines. If the user wants to put results into a knowledge graph, suggest **biotope** (separate plugin) after they confirm that direction.
