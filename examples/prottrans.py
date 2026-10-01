"""Embed one example protein sequence with ProtTrans."""

import os

import vhmodels


runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
load_kwargs = {
    "project": "prottrans",
    "model": "prot_t5_xxl_bfd",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-prottrans.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    embedding = model.embed(input=["PRTEINO"])

print("\nEmbedding:\n\n", embedding)
