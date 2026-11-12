# Model images (OCI)

Every model has an OCI image on GHCR, built from [`vhmodels/envs/Dockerfile`](../vhmodels/envs/Dockerfile) by [`.github/workflows/images.yaml`](../.github/workflows/images.yaml). The same image serves Slurm (converted to `.sif`) and Kubernetes (run directly). The model-free head image runs the Ray head on Kubernetes.

## Image contents

| Path | Content |
| --- | --- |
| `/opt/venv` | Python `MODEL_PYTHON`, the model's `runtimes.uv` requirements and `ray[serve]==RAY_VERSION` (`vhmodels/utils/uv_utils.py`) |
| `/opt/vhmodels-src` | The vhmodels package (`PYTHONPATH`) |

- Image: `ghcr.io/helmholtzai-consultants-munich/vhmodels-<model id>`
- Head image: `ghcr.io/helmholtzai-consultants-munich/vhmodels-head`, the same without model requirements (no `MODEL_DIR`, `REQUIREMENTS_FILE`)
- Build args come from `vh-checker image-spec [<model id> ...]`, which prints each model's image name, version tag and build args as JSON; `--head` prints the head image's.

## CI workflow

| Trigger | Models | Pushed tags |
| --- | --- | --- |
| Pull request | changed | none (build and smoke test only) |
| Push to `ray` | changed, limited to `mole` | `sha-<short sha>`, branch name |
| Push to `main` | changed | `sha-<short sha>`, `<model version>-ray<Ray version>` (head: `ray<Ray version>`), `latest` |
| Manual (`workflow_dispatch`) | listed, or all | as for the branch it runs on |

- **Changed models.** A change under `vhmodels/models/<Model>/` rebuilds only that model. Any other change under `vhmodels/`, `.dockerignore` or the workflow rebuilds all of them.
- **Head image.** Built in every run, so its vhmodels copy has every model's manifest.
- **Smoke test** (no GPU on GitHub runners): as a non-root user, import the model module and the Ray Serve deployment, check the Ray version, and check that torch is a CUDA build (head: Serve deployment and Ray version only). Tags are pushed only if it passes.
- **Layer cache:** `<image>:buildcache` on GHCR.
- **GPU inference** is not tested in CI. Check it on Slurm with a pulled image.

## Using an image

| Target | How |
| --- | --- |
| Slurm | `sbatch hpc-job-submit/pull_image.sbatch /abs/config.json` writes `<IMAGE_DIR>/vhmodels-<MODEL>.sif` (tag: `IMAGE_TAG`, default `latest`) |
| Kubernetes | Run it as a Ray worker pod labeled `vhmodels/project=<model id>` ([k8s/README.md](../k8s/README.md), [ray.md](ray.md#kubernetes)) |
| Local (Ray runtime only) | `apptainer pull vhmodels-<model id>.sif docker://ghcr.io/helmholtzai-consultants-munich/vhmodels-<model id>:<tag>` |

- **Both runtimes.** A plain `apptainer pull` has no `%startscript`, which `runtime="apptainer"` starts. `pull_image.sbatch` therefore builds the `.sif` from the OCI image plus the worker `%startscript`, so it serves both the [Ray runtime](ray.md#model-images) and `runtime="apptainer"`.
- **`build_apptainer.sbatch`** still builds the same image locally from `vhmodels/envs/Apptainer`, without GHCR.
- Private packages need credentials: `APPTAINER_DOCKER_USERNAME` / `APPTAINER_DOCKER_PASSWORD` (a PAT with `read:packages`) on Slurm, and an `imagePullSecret` on Kubernetes.
