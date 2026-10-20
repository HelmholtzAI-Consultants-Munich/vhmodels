"""Embed example molecules and peptides with every Hyformer variant.

The molecule variants take SMILES strings, the peptide variants amino acid
sequences. Runs locally from the repository root or through the HPC Slurm
scripts, which set VHMODELS_RUNTIME, VHMODELS_IMAGE_PATH, VHMODELS_DEVICE,
VHMODELS_DATA_DIR and VHMODELS_OUTPUT_DIR from the job config.
"""

import json
import os
from pathlib import Path

import vhmodels


default_data_dir = Path(__file__).resolve().parent.parent / "example_data"
data_dir = Path(os.environ.get("VHMODELS_DATA_DIR", str(default_data_dir))) / "Hyformer"
output_dir = Path(os.environ.get("VHMODELS_OUTPUT_DIR", "output"))


def read_lines(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


molecules = read_lines(data_dir / "molecules.smi")
peptides = read_lines(data_dir / "peptides.txt")

variants = {
    "hyformer_molecules_8M": molecules,
    "hyformer_molecules_50M": molecules,
    "hyformer_peptides_34M": peptides,
    "hyformer_peptides_34M_MIC": peptides,
}

runtime = os.environ.get("VHMODELS_RUNTIME", "conda")
load_kwargs = {
    "project": "hyformer",
    "runtime": runtime,
    "device": os.environ.get("VHMODELS_DEVICE", "auto"),
}
if runtime == "apptainer" or (runtime == "ray" and "VHMODELS_IMAGE_PATH" in os.environ):
    load_kwargs["image_path"] = os.environ.get(
        "VHMODELS_IMAGE_PATH", "vhmodels-hyformer.sif"
    )

output_dir.mkdir(parents=True, exist_ok=True)
for variant, inputs in variants.items():
    with vhmodels.load_model(model=variant, **load_kwargs) as model:
        embeddings = model.embed(input=inputs, batch_size=len(inputs))

    output_path = output_dir / f"{variant}_embeddings.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(zip(inputs, embeddings)), output_file)
    print(
        f"{variant}: {len(embeddings)} embeddings of dim {len(embeddings[0])} "
        f"-> {output_path}"
    )
