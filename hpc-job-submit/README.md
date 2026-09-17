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
- `convert_safetensors.sbatch` runs Hugging Face's official converter on a CPU node and opens a conversion PR.

Submit the jobs from the repository root:

```bash
mkdir -p output
sbatch hpc-job-submit/build_apptainer.sbatch

# After the build finishes, choose one:
sbatch hpc-job-submit/run_inference.sbatch
sbatch hpc-job-submit/run_inference_gpu.sbatch
```

The conversion job defaults to `virtual-human-chc/prot_t5_xxl_bfd` on `main`. It requires a Hugging Face login that can open a PR on the model repository. To target another model or revision:

```bash
MODEL_ID=organization/model SOURCE_REVISION=main \
    sbatch hpc-job-submit/convert_safetensors.sbatch
```

The cluster must provide `jq`, `/usr/bin/python3`, and `/usr/bin/apptainer`. Adjust the `#SBATCH` partition, QoS, time, memory, or GPU settings if required by your cluster.
