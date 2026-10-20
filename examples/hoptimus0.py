"""Embed one example histology tile with H-Optimus-0.

Runs locally from the repository root or through the HPC Slurm scripts, which
set VHMODELS_RUNTIME, VHMODELS_IMAGE_PATH, VHMODELS_DEVICE, VHMODELS_DATA_DIR
and VHMODELS_OUTPUT_DIR from the job config.
"""

import os
from pathlib import Path

import vhmodels


default_data_dir = Path(__file__).resolve().parent.parent / "example_data"
data_dir = Path(os.environ.get("VHMODELS_DATA_DIR", str(default_data_dir)))
output_dir = Path(os.environ.get("VHMODELS_OUTPUT_DIR", "output"))

runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
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
        input=str(data_dir / "H-Optimus-0" / "TUM-AACHEHDV.tif"),
        batch_size=1,
    )

output_path = output_dir / "hoptimus0_embeddings.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as output_file:
    for index, embedding in enumerate(embeddings, start=1):
        output_file.write(f"embedding{index}:\n\n{embedding}\n\n")
print(f"Wrote embeddings to {output_path}.")
