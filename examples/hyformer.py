"""Embed example molecules (SMILES) with Hyformer."""

import os

import vhmodels


runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
load_kwargs = {
    "project": "hyformer",
    "model": "hyformer_molecules_50M",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-hyformer.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    embedding = model.embed(
        input=[
            "Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1",
            "O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1",
        ]
    )

print("\nEmbedding:\n\n", embedding)
