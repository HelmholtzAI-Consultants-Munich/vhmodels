# from vhmodels.models import DinoBloom

# model = DinoBloom()
# model.load_model(variant='s', device='cpu')
# inputs = model.preprocess('path/to/image') # here preproces will be able to take single image, multiple images or path to image/images 
# print(model.embed(inputs, device='cpu'))

# import kipoi

# def main():
#     print("Kipoi loaded successfully!")
#     print(kipoi.get_model("rbp_eclip/XRCC6"))

# if __name__ == "__main__":
#     main()

# import vhmodels

# model = vhmodels.load_model(project='hyformer', model='hyformer_molecules_50M')
# results = model.transform(input=[
#         "CCCOc1cccc(-c2nn(-c3ccccc3)cc2/C=C(/C#N)C2=[N+]c3ccccc3[N-]2)c1 O=C(c1ccccc1)c1cc([N+](=O)O)c(Sc2c([N+](=O)O)cc([N+](=O)O)cc2[N+](=O)O)cc1[N+](=O)O", 
#         #"Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1 O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1",
#         #"O=C(c1cccc(/N=C(\O)CCc2ccccc2)c1)[N+]1CCCCC1"
#     ])
# print(results)

# import vhmodels

# model = vhmodels.load_model(project='dinobloom', model='s', runtime='docker')
# result = model.transform(input='example_data/DinoBloom/001.bmp')
# print(result)

# import vhmodels

# model = vhmodels.load_model(project='prottrans', model='prot_t5_xl_uniref50')

# results = model.transform(input=[
#         "PRTEINO", "SEQWENCE"
#     ])
# print(results)

import vhmodels

model = vhmodels.load_model(project='mole')
results = model.transform(input='example_data/MolE/examples_molecules.tsv')
print(results)