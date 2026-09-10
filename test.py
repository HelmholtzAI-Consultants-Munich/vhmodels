# import vhmodels

# model = vhmodels.load_model(project='hyformer', model='hyformer_molecules_50M')
# results = model.embed(input=[
#         "CCCOc1cccc(-c2nn(-c3ccccc3)cc2/C=C(/C#N)C2=[N+]c3ccccc3[N-]2)c1 O=C(c1ccccc1)c1cc([N+](=O)O)c(Sc2c([N+](=O)O)cc([N+](=O)O)cc2[N+](=O)O)cc1[N+](=O)O",
#         #"Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1 O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1",
#         #"O=C(c1cccc(/N=C(\O)CCc2ccccc2)c1)[N+]1CCCCC1"
#     ])
# print(results)

# import vhmodels

# model = vhmodels.load_model(project='dinobloom', model='s')
# result = model.embed(input='example_data/DinoBloom/001.bmp')
# print(result)

import os
import vhmodels
from pathlib import Path

with vhmodels.load_model(
    project="nicheformer",
    runtime="apptainer",
    device=os.environ.get("VHMODELS_DEVICE", "auto"),
) as model:
    embedding = model.embed(
        input={
            "technology_mean": "example_data/Nicheformer/xenium_mean_script.npy",
            "data": "example_data/Nicheformer/preprocessed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_sample-1000.h5ad",
        },
        batch_size=4,
        max_cells=4
    )

output_path = Path("output/nicheformer_embeddings.txt")

with output_path.open("w", encoding="utf-8") as output_file:
    for index, embedding in enumerate(embedding, start=1):
        output_file.write(f"embedding{index}:\n\n{embedding}\n\n")
print(f"Wrote embeddings to {output_path}.")
