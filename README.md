# Virtual-Human-CHC

This repository contains the package 'virtual-human-chc' (currently named vhmodels).

## Create a tree which explains the structure/folders in the package

## How to install the package?

## How to use the CLI?

## Quick start
Simple examples how the models can be used.

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
results = model.transform(input='example_data/MolE/examples_molecules.tsv')
print(results)
```

## TODO
1. Create folder 'notebooks' with Jupyter Notebooks showing how to use the package 
2. Create Docker file (with miniconda or mamba/minimamba) and Singularity file 
3. Create documentation (see Kipoi)
4. Extend the package with other useful functions (predict, generate etc.)
5. In the CLI, add option for the user to give you the path to the place where conda environments are stored. This will allow faster switches between environments
6. Add the rest of the models