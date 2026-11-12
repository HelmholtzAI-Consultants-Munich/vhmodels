# HPC job submission

Slurm jobs for the Apptainer and Ray runtimes. Submit them from any directory.

- **Python:** `vhmodels[cli]` in an env visible on compute nodes; the Ray jobs need the Python 3.11 driver env with `vhmodels[cli,ray]`. The jobs use `python3` from `PATH`, or `VHMODELS_PYTHON`.
- **Cluster tools:** `sbatch`, `jq`, `/usr/bin/apptainer`; `uv` for `ray/build_env.sbatch`.
- **Paths:** the config, inference script and inputs must be visible on compute nodes.

| Job | Node | Does |
| --- | --- | --- |
| `pull_image.sbatch` | CPU | Downloads the model's image from GHCR into `<IMAGE_DIR>/vhmodels-<MODEL>.sif` ([docs/images.md](../docs/images.md)) |
| `build_image.sbatch` | CPU | Builds the same `.sif` locally from `vhmodels/envs/Apptainer`; keeps an existing image |
| `apptainer/run_inference.sbatch` | CPU | Runs `INFERENCE_SCRIPT` with `runtime="apptainer"` |
| `apptainer/run_inference_gpu.sbatch` | 1 GPU | Same, on `cuda:0` |
| `ray/build_env.sbatch` | CPU | Builds a model's uv venv under `VENV_DIR` (`<MODEL>` or `--all`) |
| `ray/cluster.sbatch` | 2 GPUs on a shared node, 30 min | Single-node Ray cluster with Ray Serve; deploys an optional Serve config (2nd argument) and stays up until `scancel` or the time limit |

## Configuration

Copy a template anywhere, fill in absolute paths visible on the compute nodes, and pass its absolute path as the first argument of every job.

| Template | Describes |
| --- | --- |
| [`apptainer/config.json`](apptainer/config.json) | One inference run (model + script); for example one config per model |
| [`ray/config.json`](ray/config.json) | One cluster; models are chosen per Ray Job or Serve app |

| Key | Apptainer | Ray | Purpose |
| --- | --- | --- | --- |
| `IMAGE_DIR` | ✓ | ✓ | `.sif` images; Ray actors run in them when set |
| `APPTAINER_CACHEDIR` | ✓ | ✓ | Apptainer's host cache for base image layers |
| `UV_CACHE_DIR` | ✓ | ✓ | uv cache for packages and the managed Python |
| `IMAGE_TAG` | ✓ | ✓ | GHCR tag for `pull_image.sbatch`; default `latest` |
| `HF_CACHE_DIR` | ✓ | ✓ | Hugging Face cache; weights go to `<HF_CACHE_DIR>/hub`, the login token stays in `~/.cache/huggingface` |
| `TORCH_CACHE_DIR` | ✓ | ✓ | PyTorch model cache |
| `DATA_DIR` | ✓ | ✓ | Optional input directory, mounted into the images |
| `OUTPUT_DIR` | ✓ | ✓ | Optional result directory; default `output/` beside the config |
| `MODEL` | ✓ | | Registered model ID; default model of the image jobs |
| `INFERENCE_SCRIPT` | ✓ | | Python script to run |
| `VENV_DIR` | | ✓ | One uv venv per model, used when `IMAGE_DIR` is unset |

## Images

Both runtimes run models from `<IMAGE_DIR>/vhmodels-<MODEL>.sif`. The image jobs take either config; the model is the optional 2nd argument, otherwise the config's `MODEL`.

```bash
HPC_SCRIPTS=/path/to/vhmodels/slurm
sbatch "${HPC_SCRIPTS}/pull_image.sbatch" /abs/apptainer-config.json      # MODEL from the config
sbatch "${HPC_SCRIPTS}/pull_image.sbatch" /abs/ray-config.json hyformer   # one job per model
```

## Apptainer runtime

- The jobs export `VHMODELS_RUNTIME=apptainer`, `VHMODELS_IMAGE_PATH`, `VHMODELS_DATA_DIR` and `VHMODELS_OUTPUT_DIR` (plus `VHMODELS_DEVICE=cuda:0` on GPU).
- The scripts in [`examples/`](../examples/) read these, so any of them works as `INFERENCE_SCRIPT` with `DATA_DIR` set to the repository's `example_data/`.

```bash
CONFIG=/abs/path/apptainer-config.json
sbatch "${HPC_SCRIPTS}/pull_image.sbatch" "${CONFIG}"   # or build_image.sbatch

# After the image job finishes, choose one:
sbatch "${HPC_SCRIPTS}/apptainer/run_inference.sbatch" "${CONFIG}"
sbatch "${HPC_SCRIPTS}/apptainer/run_inference_gpu.sbatch" "${CONFIG}"
```

## Ray runtime (experimental)

- Set `VHMODELS_PYTHON` to the driver env's Python.
- Actors run in the images in `IMAGE_DIR`, or in the venvs under `VENV_DIR` if `IMAGE_DIR` is unset.
- Ray Jobs and Ray Serve on the cluster: [docs/ray.md](../docs/ray.md#slurm-cluster).

```bash
CONFIG=/abs/path/ray-config.json
sbatch "${HPC_SCRIPTS}/pull_image.sbatch" "${CONFIG}" hyformer   # or: ray/build_env.sbatch "${CONFIG}" hyformer
sbatch "${HPC_SCRIPTS}/ray/cluster.sbatch" "${CONFIG}" /abs/path/serve.yaml   # Serve config optional
```

Slurm logs go to `<job-name>-<job-id>.out` and `.err` in the submission directory. Adjust the `#SBATCH` partition, QoS, time, memory or GPU settings for your cluster.
