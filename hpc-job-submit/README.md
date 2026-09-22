# HPC job submission

These repository examples build an Apptainer image and run a Python inference script on
an HPC node. You can submit them from any directory. They are not part of the
installed `vhmodels` package. The build uses model files from that package.

Install `vhmodels[cli]` in a Python environment visible on compute nodes. The
configuration, inference script, and its input files must also be visible there. The
cluster must provide `sbatch`, `jq`, and `/usr/bin/apptainer`.

## Configuration

Copy [`config.json`](config.json) to `config.local.json`. Fill in the required
paths with absolute paths available on the compute nodes. The local file is
ignored by Git.

| Setting | Purpose |
| --- | --- |
| `IMAGE_DIR` | Stores the built `.sif` image |
| `APPTAINER_CACHEDIR` | Host cache for the Ubuntu and uv base image layers |
| `UV_CACHE_DIR` | Host cache for Python packages and the managed Python used during image builds |
| `HF_CACHE_DIR` | Hugging Face model cache |
| `TORCH_CACHE_DIR` | PyTorch model cache |
| `MODEL` | Registered model ID |
| `INFERENCE_SCRIPT` | Python script to run for inference |
| `DATA_DIR` | Optional data directory passed to the inference script and mounted into Apptainer |
| `OUTPUT_DIR` | Optional result directory; defaults to `output/` beside the config file |

The build mounts `UV_CACHE_DIR` at `/opt/uv-cache` for uv inside
Apptainer. `APPTAINER_CACHEDIR` stays on the host and is used by Apptainer
itself. These are separate caches with separate contents.

## Submit

The config supplies the inference script, data, and output paths. Activate the Python
environment with `vhmodels[cli]` installed before submitting. The jobs use
`python3` from `PATH`; set `VHMODELS_PYTHON` if a different interpreter is
needed. Submit from any directory with the sbatch script and the config path:

```bash
HPC_SCRIPTS=/path/to/vhmodels/hpc-job-submit
CONFIG="${HPC_SCRIPTS}/config.local.json"

sbatch "${HPC_SCRIPTS}/build_apptainer.sbatch" "${CONFIG}"

# After the build finishes, choose one:
sbatch "${HPC_SCRIPTS}/run_inference.sbatch" "${CONFIG}"
sbatch "${HPC_SCRIPTS}/run_inference_gpu.sbatch" "${CONFIG}"
```

Inference runs with `OUTPUT_DIR` as its working directory. The inference script,
output, configured data directory, and model caches are mounted into the
Apptainer worker. Set `APPTAINER_BINDPATH` before submission if the inference script
needs other directories. Slurm writes `%x-%j.out` and `%x-%j.err` logs in the
directory where you run `sbatch`; these logs are opened before the job script
changes directory.

The repository's [`test.py`](../test.py) is an example inference script for Nicheformer.
Set `MODEL=nicheformer`. `test.py` defaults to the repository's `example_data/`;
set `DATA_DIR` to another directory containing `Nicheformer/` to override it.
The job passes `DATA_DIR` and `OUTPUT_DIR` to the inference script as
`VHMODELS_DATA_DIR` and `VHMODELS_OUTPUT_DIR`. For a different model or a wheel
installation, provide your own inference script. Direct runs of `test.py`
still write to `output/` relative to the current directory.

The build script uses the selected Python interpreter to run the package's
Apptainer image builder. The inference scripts run the configured script with the same
interpreter; Python finds `vhmodels` on its normal import path. Each inference
job prints the package path it imported.

Adjust the `#SBATCH` partition, QoS, time, memory, or GPU settings in the
templates for your cluster, or override them with `sbatch` options.
