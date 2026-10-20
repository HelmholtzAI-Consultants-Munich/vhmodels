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
| `IMAGE_DIR` | Stores the built `.sif` image; with the Ray runtime, actors run in it when set |
| `APPTAINER_CACHEDIR` | Host cache for the Ubuntu and uv base image layers |
| `UV_CACHE_DIR` | Host cache for Python packages and the managed Python used during image builds |
| `VENV_DIR` | Ray runtime without `IMAGE_DIR`: stores one uv venv per model |
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

## Ray runtime (experimental)

Instead of an Apptainer image, the [Ray runtime](../docs/ray.md) runs the model
as a Ray actor in a uv venv under `VENV_DIR`. Both jobs need `uv` in `PATH` and
`VHMODELS_PYTHON` pointing at the Python 3.11 driver environment with
`vhmodels[cli,ray]` installed, because the venv must match its Python and Ray
versions:

```bash
export VHMODELS_PYTHON=/path/to/vhmodels/.venv/bin/python

sbatch "${HPC_SCRIPTS}/build_ray_env.sbatch" "${CONFIG}"
# After the build finishes:
sbatch "${HPC_SCRIPTS}/run_inference_ray.sbatch" "${CONFIG}"
```

- `build_ray_env.sbatch` runs `vh-checker create-ray-env` on a CPU node. It reuses an up-to-date venv.
- `run_inference_ray.sbatch` starts a single-node Ray head inside the GPU allocation. Its ports are derived from the job id and its temp dir is under `/localscratch`. It then runs the inference script with `RAY_ADDRESS`, `VHMODELS_RUNTIME=ray` and `VHMODELS_DEVICE=cuda:0` set. With `IMAGE_DIR`, it also sets `VHMODELS_IMAGE_PATH` and the same bind mounts as `run_inference.sbatch`, and the actor runs in the image from `build_apptainer.sbatch`; otherwise it sets `VHMODELS_VENV_DIR` and needs no bind mounts. Slurm stops the Ray cluster when the job ends.

The conversion job defaults to `virtual-human-chc/prot_t5_xxl_bfd` on `main`. It requires a Hugging Face login that can open a PR on the model repository. To target another model or revision:

```bash
MODEL_ID=organization/model SOURCE_REVISION=main \
    sbatch "${HPC_SCRIPTS}/convert_safetensors.sbatch" "${CONFIG}"
```

Slurm logs are written to `<job-name>-<job-id>.out` and `.err` in the submission
directory. Adjust the `#SBATCH` partition, QoS, time, memory, or GPU settings if
required by your cluster.
