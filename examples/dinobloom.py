"""Embed one example blood cell image with DinoBloom."""

import os

import vhmodels


runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
load_kwargs = {
    "project": "dinobloom",
    "model": "s",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-dinobloom.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    embedding = model.embed(input="example_data/DinoBloom/001.bmp")

print("\nEmbedding:\n\n", embedding)
