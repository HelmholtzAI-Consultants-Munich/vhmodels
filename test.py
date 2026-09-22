"""Smoke-test H-Optimus-0 on the example histology tile."""

import os
from pathlib import Path

import vhmodels


runtime = os.environ.get("VHMODELS_RUNTIME", "apptainer")
load_kwargs = {
    "project": "hoptimus0",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-hoptimus0.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    embeddings = model.embed(
        input="example_data/H-Optimus-0/TUM-AACHEHDV.tif",
        batch_size=1,
    )

output_path = Path("output/hoptimus0_embeddings.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as output_file:
    for index, embedding in enumerate(embeddings, start=1):
        output_file.write(f"embedding{index}:\n\n{embedding}\n\n")
print(f"Wrote embeddings to {output_path}.")
