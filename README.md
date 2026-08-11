# Virtual-Human-CHC

This repository contains the vhmodels package. Its goal is to unify access to multiple deep learning models developed at Helmholtz Munich.

## Core concept

Each model lives in its own directory with its implementation, metadata, and
runtime dependencies. `vhmodels` executes it in either a per-model Conda
environment or an Apptainer container. Apptainer images use Ubuntu 24.04 and a
uv-managed Python environment at `/opt/venv`.

## Installation

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/vhmodels.git
cd vhmodels
pip install -e ".[cli]"
```

For Conda execution, install [Miniforge](https://conda-forge.org/download/). For
Apptainer execution, install Apptainer on Linux/HPC or Lima on macOS
(`brew install lima`); `vhmodels` manages the Lima VM automatically. You do not
need to install uv on the host.

## CLI usage

### List available models

```bash
vh-checker list
```

### Create a Conda environment

```bash
vh-checker create-env <model_name>
```

Example:

```bash
vh-checker create-env dinobloom
```

### Create an Apptainer image

```bash
vh-checker create-apptainer-image <model_name>
```

By default this creates `vhmodels-<model_name>.sif` in the current directory. Run a model from that image with e.g.:

```python
import vhmodels

model = vhmodels.load_model(
    project="dinobloom", model="s", runtime="apptainer"
)
```

Pass `image_path="/path/to/image.sif"` to `load_model` when the image is stored
elsewhere. The CLI equivalent is
`vh-checker run ... --runtime apptainer --image-path /path/to/image.sif`.
Set `APPTAINER_NV=1` when running directly on an NVIDIA Linux/HPC host to expose
its GPU to the container. The Lima path on macOS is CPU-only.

## Quick start

First create the corresponding Conda environment or Apptainer image. The
examples below use the default Conda runtime; pass `runtime="apptainer"` to use
an image.

### DinoBloom

```python
import vhmodels

model = vhmodels.load_model(project='dinobloom', model='s')
results = model.embed(input='example_data/DinoBloom/001.bmp')
print(results)
```

### Hyformer

```python
import vhmodels

model = vhmodels.load_model(project='hyformer', model='hyformer_molecules_50M')
results = model.embed(input=[
        "CCCOc1cccc(-c2nn(-c3ccccc3)cc2/C=C(/C#N)C2=[N+]c3ccccc3[N-]2)c1 O=C(c1ccccc1)c1cc([N+](=O)O)c(Sc2c([N+](=O)O)cc([N+](=O)O)cc2[N+](=O)O)cc1[N+](=O)O",
        "Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1 O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1",
        "O=C(c1cccc(/N=C(\O)CCc2ccccc2)c1)[N+]1CCCCC1"
    ])
print(results)
```

### ProtTrans

```python
import vhmodels

model = vhmodels.load_model(project='prottrans', model='prot_t5_xl_uniref50')

results = model.embed(input=[
        "PRTEINO", "SEQWENCE"
    ])
print(results)
```

### MolE

```python
import vhmodels

model = vhmodels.load_model(project='mole')
results = model.embed(input='example_data/MolE/sequences.smiles')
print(results)
```

## Adding models
If you want to add your own model to `vhmodels`, please follow our [contribution guideline](model_contribution.md).

## Installing AI skills

Use your IDE's plugin or marketplace interface, or copy the contents of the
`skills` directory into the corresponding project directory (for example,
`.cursor/skills/` for Cursor).

For more details:

- [claude code](https://code.claude.com/docs/en/skills)
- [cursor](https://cursor.com/docs/skills)
- [codex](https://developers.openai.com/codex/skills)

## Folder structure

```text
vhmodels
├───example_data # Example inputs/outputs (also available on Hugging Face)
│   ├───DinoBloom
│   └───MolE
├───notebooks # Example notebooks demonstrating usage
├───tests # Tests for core model functionality
└───vhmodels
    ├───envs # Template for Apptainer environments
    ├───models # Implementations + Conda/uv dependencies + metadata
    │   ├───DinoBloom
    │   ├───Hyformer
    │   ├───MolE
    │   └───ProtTrans
    ├───utils # Utility functions
    └───vh_checker # CLI + base model interface
```
