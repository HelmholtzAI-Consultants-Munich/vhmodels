# HPC job submission

These Slurm scripts build an Apptainer image and run `test.py` on an HPC node.

## Configuration

Edit `hpc-job-submit/config.json` before submitting a job. All directory paths should be absolute, writable, and available on the compute nodes.

| Setting | Purpose |
| --- | --- |
| `IMAGE_DIR` | Stores the built `.sif` image |
| `APPTAINER_CACHEDIR` | Apptainer build cache |
| `UV_CACHE_DIR` | Python package cache used during the build |
| `HF_CACHE_DIR` | Hugging Face model cache |
| `TORCH_CACHE_DIR` | PyTorch model cache |
| `MODEL` | Registered model ID, such as `nicheformer` |

## Scripts

- `build_apptainer.sbatch` builds the image for the configured model.
- `run_inference.sbatch` runs CPU inference.
- `run_inference_gpu.sbatch` runs inference on one NVIDIA GPU.

Submit the jobs from the repository root:

```bash
mkdir -p output
sbatch hpc-job-submit/build_apptainer.sbatch

# After the build finishes, choose one:
sbatch hpc-job-submit/run_inference.sbatch
sbatch hpc-job-submit/run_inference_gpu.sbatch
```

The cluster must provide `jq`, `/usr/bin/python3`, and `/usr/bin/apptainer`. Adjust the `#SBATCH` partition, QoS, time, memory, or GPU settings if required by your cluster.
