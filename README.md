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

# loads the model from HuggingFace repository
model = vhmodels.load_model(project='dinobloom', model='s')

# creates the embedings
# 'transform' can take either path to image/folder of images 
# or image/s directly 
result = model.transform({'inputs': 'vhmodels/models/DinoBloom/001.bmp'})

print(result)
#Example output (truncated):
#{'output': [[-1.133154034614563, 0.8950058221817017, -2.936743974685669, -4.055114269256592, 0.45779430866241455, 0.5101346373558044, 3.2842488288879395, 0.39938172698020935, 2.283127784729004, 2.921227216720581, -2.974207878112793, -8.045597076416016, -0.13773316144943237, 5.404041767120361, -1.6217663288116455, 0.9758090972900391, -2.1749391555786133]]}
```

### Hyformer

```python
import vhmodels

model = vhmodels.load_model(project='hyformer', model='hyformer_molecules_50M', device='cpu')

result = model.transform({'inputs': [
        'CCCOc1cccc(-c2nn(-c3ccccc3)cc2/C=C(/C#N)C2=[N+]c3ccccc3[N-]2)c1 O=C(c1ccccc1)c1cc([N+](=O)O)c(Sc2c([N+](=O)O)cc([N+](=O)O)cc2[N+](=O)O)cc1[N+](=O)O' ,
        'Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1 O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1',
        'O=C(c1cccc(/N=C(\O)CCc2ccccc2)c1)[N+]1CCCCC1'
    ]}, 
    batch_size=128, device='cpu')
print(result)

# Example output (truncated):
# {'output': [[-1.133154034614563, 0.8950058221817017, -2.936743974685669, -4.055114269256592, 0.45779430866241455, 0.5101346373558044, 3.2842488288879395, 0.39938172698020935, 2.283127784729004, 2.921227216720581, -2.974207878112793, -8.045597076416016, -0.13773316144943237, 5.404041767120361, -1.6217663288116455, 0.9758090972900391, -2.1749391555786133]]}
```

### ProtTrans

```python
import vhmodels

model = vhmodels.load_model(project='prottrans', model='prot_t5_xl_uniref50', device='cpu')
result = model.transform({'inputs': ["PRTEINO", "SEQWENCE"]}) # possibly also select device here / make transform accept also preprocess sequences?
print(result)
```

### MolE

```python
import vhmodels

model = vhmodels.load_model(project='mole', device='cpu')
result = model.transform({'inputs':
                            {
                                'smiles': 'input/examples_molecules.tsv',
                                'screening_results': 'input/maier_screening_results.tsv.gz'
                            }}, device='cpu')
print(result)
```

## TODO
1. Create folder 'notebooks' with Jupyter Notebooks showing how to use the package 
2. Create Docker file (with miniconda or mamba/minimamba) and Singularity file 
3. Create documentation (see Kipoi)
4. Extend the package with other useful functions (predict, generate etc.)
5. In the CLI, add option for the user to give you the path to the place where conda environments are stored. This will allow faster switches between environments
6. Add the rest of the models