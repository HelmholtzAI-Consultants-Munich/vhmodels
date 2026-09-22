"""Predict antimicrobial activity for example molecules with MolE.

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
    "project": "mole",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer":
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-mole.sif"
    )

with vhmodels.load_model(**load_kwargs) as model:
    results = model.predict(
        input=str(data_dir / "MolE" / "examples_molecules.tsv")
    )

output_path = output_dir / "mole_predictions.tsv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as output_file:
    output_file.write("pred_id\tantimicrobial_predictive_probability\n")
    for pred_id, probability in results.items():
        output_file.write(f"{pred_id}\t{probability}\n")
print(f"Wrote predictions to {output_path}.")
