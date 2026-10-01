"""Predict antimicrobial activity for example molecules with MolE."""

import os

import vhmodels


runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
load_kwargs = {
    "project": "mole",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-mole.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    results = model.predict(input="example_data/MolE/examples_molecules.tsv")

print("\nPrediction:\n\n", results)
