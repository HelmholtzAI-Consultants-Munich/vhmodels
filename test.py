"""Run inference examples with configurable data and output paths."""

import os
import vhmodels
from pathlib import Path

default_data_dir = Path(__file__).resolve().parent / "example_data"
data_dir = Path(os.environ.get("VHMODELS_DATA_DIR", str(default_data_dir))) / "Nicheformer"
output_dir = Path(os.environ.get("VHMODELS_OUTPUT_DIR", "output"))

with vhmodels.load_model(
    project="nicheformer",
    runtime="apptainer",
    image_path=os.environ.get("VHMODELS_IMAGE_PATH", "vhmodels-nicheformer.sif"),
    device=os.environ.get("VHMODELS_DEVICE", "auto"),
) as model:
    embedding = model.embed(
        input={
            "technology_mean": str(data_dir / "xenium_mean_script.npy"),
            "data": str(data_dir / "preprocessed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_sample-1000.h5ad"),
        },
        batch_size=4,
        max_cells=4
    )

output_path = output_dir / "nicheformer_embeddings.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output_file:
    for index, embedding in enumerate(embedding, start=1):
        output_file.write(f"embedding{index}:\n\n{embedding}\n\n")
print(f"Wrote embeddings to {output_path}.")
