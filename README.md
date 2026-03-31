# Virtual-Human-CHC

This repository contains the package 'virtual-human-chc' (currently named vhmodels). Its goal is to unify and orchestrate multiple deep learning models developed at Helmholtz Munich - without tightly coupling or reimplementing them.

## Core concept
Each model Lives in its own directory, has its own Conda environment, Docker image, or Singularity image and is executed via a subprocess.

## Folder structure
```
virtual_human_chc
├───example_data # Example inputs/outputs (also available on HuggingFace)
│   ├───DinoBloom
│   └───MolE
├───notebooks # Example notebooks demonstrating usage
├───tests # Tests for core model functionality
└───vhmodels
    ├───envs # Templates for Docker and Singularity environments
    ├───models # Model implementations + environments + metadata
    │   ├───DinoBloom
    │   ├───Hyformer
    │   ├───MolE
    │   └───ProtTrans
    ├───utils # Utility functions
    └───vh_checker # CLI + base model interface
```

## Installation

From the root directory (```virtual_human_chc```), run:

```
pip install -e .
```

## CLI Usage

### List available models:
```vh-checker list```

### Create ```conda``` environment:
```vh-checker create-env <model_name>```

Example: ```vh-checker create-env dinobloom```

### Create ```Docker``` image:
```vh-checker create-docker-image <model_name>```

Example: ```vh-checker create-docker-image dinobloom```


## Quick start
Simple examples how the models can be used. To use the models, you should first create the corresponding ```conda```/```Docker``` environment.

### DinoBloom

```python
import vhmodels

model = vhmodels.load_model(project='dinobloom', model='s')
result = model.transform(input='example_data/DinoBloom/001.bmp')
print(result)
```

### Hyformer

```python
import vhmodels

model = vhmodels.load_model(project='hyformer', model='hyformer_molecules_50M')
results = model.transform(input=[
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

results = model.transform(input=[
        "PRTEINO", "SEQWENCE"
    ])
print(results)
```

### MolE

```python
import vhmodels

model = vhmodels.load_model(project='mole')
results = model.transform(input='example_data/MolE/sequences.smiles')
print(results)
```

## TODOs

1. Implement MolE and ProtTrans (add also predict functions)
2. Implement MolE and ProtTrans in the Docker and Singularity scheme
3. Document everything
4. In the CLI, add option for the user to give you the path to the place where conda environments are stored. This will allow faster switches between environments