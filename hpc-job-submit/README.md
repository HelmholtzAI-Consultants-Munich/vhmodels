# HPC job submission

These repository examples build an Apptainer image and run a Python inference script on
an HPC node. You can submit them from any directory. They are not part of the
installed `vhmodels` package. The build uses model files from that package.

Install `vhmodels[cli]` in a Python environment visible on compute nodes. The
configuration, inference script, and its input files must also be visible there. The
cluster must provide `sbatch`, `jq`, and `/usr/bin/apptainer`.

## Configuration

Every job takes the absolute path of a JSON config with the keys below.
[`config.json`](config.json) is a template: copy it anywhere, for example one
config per model, and fill in absolute paths available on the compute nodes.
`config.local.json` in this folder is ignored by Git for convenience.

| Setting | Purpose |
| --- | --- |
| `IMAGE_DIR` | Stores the built `.sif` image |
| `APPTAINER_CACHEDIR` | Host cache for the Ubuntu and uv base image layers |
| `UV_CACHE_DIR` | Host cache for Python packages and the managed Python used during image builds |
| `HF_CACHE_DIR` | Hugging Face model cache; weights go to `<HF_CACHE_DIR>/hub`, while the login token stays in `~/.cache/huggingface` |
| `TORCH_CACHE_DIR` | PyTorch model cache |
| `MODEL` | Registered model ID |
| `INFERENCE_SCRIPT` | Python script to run for inference |
| `DATA_DIR` | Optional data directory passed to the inference script and mounted into Apptainer |
| `OUTPUT_DIR` | Optional result directory; defaults to `output/` beside the config file |

The build mounts `UV_CACHE_DIR` at `/opt/uv-cache` for uv inside
Apptainer. `APPTAINER_CACHEDIR` stays on the host and is used by Apptainer
itself. These are separate caches with separate contents.

- `build_apptainer.sbatch` builds the image for the configured model.
- `run_inference.sbatch` runs CPU inference.
- `run_inference_gpu.sbatch` runs inference on one NVIDIA GPU.
- `convert_safetensors.sbatch` runs Hugging Face's official converter on a CPU node and opens a conversion PR.

The config supplies the inference script, data, and output paths. The
inference jobs export `VHMODELS_RUNTIME=apptainer`, `VHMODELS_IMAGE_PATH`,
`VHMODELS_DATA_DIR`, and `VHMODELS_OUTPUT_DIR` (plus `VHMODELS_DEVICE=cuda:0` on
GPU) for the script. The scripts in [`examples/`](../examples/) read these
variables, so any of them can be used as `INFERENCE_SCRIPT` with `DATA_DIR`
pointing at the repository's `example_data/`. Activate the Python
environment with `vhmodels[cli]` installed before submitting. The jobs use
`python3` from `PATH`; set `VHMODELS_PYTHON` if a different interpreter is
needed. Submit from any directory with the sbatch script and the config path:

```bash
HPC_SCRIPTS=/path/to/vhmodels/hpc-job-submit
CONFIG=/absolute/path/to/your-config.json

sbatch "${HPC_SCRIPTS}/build_apptainer.sbatch" "${CONFIG}"

# After the build finishes, choose one:
sbatch "${HPC_SCRIPTS}/run_inference.sbatch" "${CONFIG}"
sbatch "${HPC_SCRIPTS}/run_inference_gpu.sbatch" "${CONFIG}"
```

The conversion job defaults to `virtual-human-chc/prot_t5_xxl_bfd` on `main`. It requires a Hugging Face login that can open a PR on the model repository. To target another model or revision:

```bash
MODEL_ID=organization/model SOURCE_REVISION=main \
    sbatch "${HPC_SCRIPTS}/convert_safetensors.sbatch" "${CONFIG}"
```

Slurm logs are written to `<job-name>-<job-id>.out` and `.err` in the submission
directory. Adjust the `#SBATCH` partition, QoS, time, memory, or GPU settings if
required by your cluster.
