# Model images (OCI)

Every model has an image on GHCR,
`ghcr.io/helmholtzai-consultants-munich/vhmodels-<model id>`, with the model's
environment and the vhmodels package. The same image runs on Slurm (as `.sif`)
and on Kubernetes. `vhmodels-head` is a model-free image for the Ray head on
Kubernetes.

## Tags

| Tag | Image |
| --- | --- |
| `latest` | latest build from `main` |
| `<model version>-ray<Ray version>` | a model version from `main` (head: `ray<Ray version>`) |
| `sha-<short commit>` | one commit; never changes |
| `ray` | latest build from the experimental `ray` branch (MolE and head only) |

## Using an image

| Target | How |
| --- | --- |
| Slurm | `sbatch hpc-job-submit/pull_image.sbatch /abs/config.json` writes `<IMAGE_DIR>/vhmodels-<MODEL>.sif` (tag: `IMAGE_TAG`, default `latest`) |
| Kubernetes | Run it as a Ray worker pod labeled `vhmodels/project=<model id>` ([k8s/README.md](../k8s/README.md)) |
| Local (Ray runtime only) | `apptainer pull vhmodels-<model id>.sif docker://ghcr.io/helmholtzai-consultants-munich/vhmodels-<model id>:<tag>` |

- **Both runtimes.** The `.sif` from `pull_image.sbatch` works with `runtime="apptainer"` and `runtime="ray"`; a plain `apptainer pull` only with `runtime="ray"`.
- **Without GHCR,** `build_apptainer.sbatch` builds the same image on the cluster.
- **Private images** need credentials: `APPTAINER_DOCKER_USERNAME` / `APPTAINER_DOCKER_PASSWORD` (a GitHub token with `read:packages`) on Slurm, and an `imagePullSecret` on Kubernetes.
