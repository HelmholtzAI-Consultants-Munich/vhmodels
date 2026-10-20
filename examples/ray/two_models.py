"""Run Hyformer and ProtTrans concurrently as Ray actors, each in its own image or venv."""

import os
from pathlib import Path

import ray

import vhmodels


default_data_dir = Path(__file__).resolve().parents[2] / "example_data"
data_dir = Path(os.environ.get("VHMODELS_DATA_DIR", str(default_data_dir)))
device = os.environ.get("VHMODELS_DEVICE", "auto")

molecules = [
    line.strip()
    for line in (data_dir / "Hyformer" / "molecules.smi").read_text().splitlines()
    if line.strip()
]
sequences = ["PRTEINO", "SEQWENCE"]


def shape(value):
    """Return the nested-list shape of a model output, e.g. '3x256'."""
    dims = []
    while isinstance(value, list):
        dims.append(len(value))
        value = value[0] if value else None
    return "x".join(map(str, dims))


hyformer = vhmodels.load_model(
    project="hyformer", model="hyformer_molecules_8M", runtime="ray", device=device
)
prottrans = vhmodels.load_model(
    project="prottrans", model="prot_bert", runtime="ray", device=device
)

with hyformer, prottrans:
    # Submit both requests before waiting, so both models run concurrently.
    pending = [
        hyformer.actor.embed.remote(molecules, batch_size=len(molecules)),
        prottrans.actor.embed.remote(sequences),
    ]
    hyformer_result, prottrans_result = ray.get(pending)

    for name, model, result in (
        ("hyformer", hyformer, hyformer_result),
        ("prottrans", prottrans, prottrans_result),
    ):
        info = model.info()
        print(
            f"{name}: output {shape(result['output'])} | "
            f"python {info['executable']} | pid {info['pid']} on {info['host']} | "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
