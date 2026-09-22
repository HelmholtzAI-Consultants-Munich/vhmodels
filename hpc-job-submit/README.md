# HPC job submission

These Slurm scripts build an Apptainer image and run the repository-level
`test.py` inference entry point on an HPC node. The included `test.py` example
uses H-Optimus-0, so its configuration uses `MODEL=hoptimus0`. It embeds
`example_data/H-Optimus-0/TUM-AACHEHDV.tif` and writes the embedding to
`output/hoptimus0_embeddings.txt`. Hugging Face access to the gated model must
be approved before running the example.

## Configuration

Edit `hpc-job-submit/config.json` before submitting a job. Replace every `null`
path with an absolute and writable directory available on your cluster. The
scripts fail immediately if a required value is still `null`.

| Setting | Purpose |
| --- | --- |
| `IMAGE_DIR` | Stores the built `.sif` image |
| `APPTAINER_CACHEDIR` | Apptainer build cache |
| `UV_CACHE_DIR` | Python package cache used during the build |
| `HF_CACHE_DIR` | Hugging Face model cache |
| `TORCH_CACHE_DIR` | PyTorch model cache |
| `MODEL` | Registered model ID; set to `hoptimus0` for the `test.py` example |

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
