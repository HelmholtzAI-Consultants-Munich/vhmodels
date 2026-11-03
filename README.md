# Virtual-Human-CHC

This repository contains the vhmodels package. Its goal is to unify access to multiple deep learning models developed at Helmholtz Munich.

## Core concept

Each model lives in its own directory with its implementation, metadata, and
runtime dependencies. `vhmodels` executes it in either a per-model Conda
environment or an Apptainer container. Apptainer images use Ubuntu 24.04 and a
uv-managed Python environment at `/opt/venv`.

Metadata and resource locations are declared in `model.json` +
`manifests/<variant>.json`, validated against a Pydantic schema and resolved
by `vhmodels.models.registry.Registry`; see [docs/manifest.md](docs/manifest.md).

## Installation

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/vhmodels.git
cd vhmodels
pip install -e ".[cli]"
```

For Conda execution, install [Miniforge](https://conda-forge.org/download/). For
Apptainer execution, install Apptainer on Linux/HPC or Lima on macOS
(`brew install lima`); `vhmodels` manages the Lima VM automatically. You do not
need to install uv on the host.

## CLI usage

### List available models

```bash
vh-checker list
```

### Create a Conda environment

```bash
vh-checker create-env <model_name>
```

Example:

```bash
vh-checker create-env dinobloom
```

### Create an Apptainer image

```bash
vh-checker create-apptainer-image <model_name>
```

By default this creates `vhmodels-<model_name>.sif` in the current directory. Run a model from that image with e.g.:

```python
import vhmodels

model = vhmodels.load_model(
    project="dinobloom", model="s", runtime="apptainer"
)
```

Pass `image_path="/path/to/image.sif"` to `load_model` when the image is stored
elsewhere. The CLI equivalent is
`vh-checker run ... --runtime apptainer --image-path /path/to/image.sif`.
Set `APPTAINER_NV=1` when running directly on an NVIDIA Linux/HPC host to expose
its GPU to the container. The Lima path on macOS is CPU-only.

### Ray runtime (experimental)

`runtime="ray"` runs the model as a Ray actor in its Apptainer image or in a
per-model uv venv, both built from the model's `runtimes.uv` spec. It needs a
Python 3.11 driver environment with `vhmodels[ray]` (and uv for venvs):

```bash
uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[cli,ray]"
.venv/bin/vh-checker create-ray-env hyformer
```

```python
model = vhmodels.load_model(
    project="hyformer", model="hyformer_molecules_8M", runtime="ray"
)
```

See [docs/ray.md](docs/ray.md) for details, Slurm usage and the roadmap.

### Model lifetime

Both Conda- and Apptainer-backed models start one worker lazily on their first
`embed()` call. The model is loaded once in that worker and reused by later
calls on the same model object. Conda uses a local process; Apptainer uses a
background instance. Call `model.close()` when finished, or use the model as a
context manager; remaining workers are also stopped when Python exits normally.

Abrupt termination may leave an Apptainer instance behind; inspect with
`apptainer instance list` and stop it with `apptainer instance stop <name>`.
Apptainer bind mounts are fixed when the instance starts, so set
`APPTAINER_BINDPATH` before the first `embed()` when inputs live outside
Apptainer's default host mounts. Rebuild images created with an older version of
`vhmodels`, since the persistent worker is installed by the image template.

```python
with vhmodels.load_model(
    project="dinobloom", model="s"
) as model:
    first = model.embed(input="first.bmp")
    second = model.embed(input="second.bmp")  # reuses the loaded model
```

Each runtime reaches the model through these classes:

| Role | Conda / Apptainer | Ray |
| --- | --- | --- |
| Model implementation | `BaseModel` subclass, `vhmodels/models/<Model>/model.py` | same |
| Handle returned by `load_model()` | `ModelProxy`, `vhmodels/vh_checker/factory.py` | `RayModelProxy`, `vhmodels/vh_ray/client.py` |
| Starts, reuses and stops the model process | `ModelProcessManager`, `vhmodels/vh_checker/process_manager.py` | Ray cluster (`ray.init`, `ray.kill`) |
| Isolated environment | `CondaBackend` / `ApptainerBackend`, `vhmodels/vh_checker/backends.py` | `runtime_env` with the venv or image Python, `vhmodels/vh_ray/envs.py` |
| Request channel | Unix socket, `vhmodels/vh_checker/transports.py` + `protocol.py` | Ray actor calls (`.remote()`) |
| Process that holds the model | `ModelWorker`, `vhmodels/vh_checker/worker.py` | `ModelActor`, `vhmodels/vh_ray/actor.py` |
| HTTP serving | — | `ModelDeployment` (Ray Serve), `vhmodels/vh_ray/serve/deployment.py` |

## Examples

Each script under [`examples/`](examples/) loads one model on its example input
from [`example_data/`](example_data/). Create the model's Conda environment or
Apptainer image first. Every script also works as `INFERENCE_SCRIPT` in the
[HPC job submission guide](hpc-job-submit/README.md).

| Model | Script |
| --- | --- |
| DinoBloom | [`examples/dinobloom.py`](examples/dinobloom.py) |
| H-Optimus-0 | [`examples/hoptimus0.py`](examples/hoptimus0.py) |
| Hyformer | [`examples/hyformer.py`](examples/hyformer.py) |
| MolE | [`examples/mole.py`](examples/mole.py) |
| Nicheformer | [`test.py`](test.py) |
| ProtTrans | [`examples/prottrans.py`](examples/prottrans.py) |
| Two models on Ray | [`examples/ray/two_models.py`](examples/ray/two_models.py) |

- **H-Optimus-0** is gated: accept the conditions on its
  [model page](https://huggingface.co/bioptimus/H-optimus-0), then run
  `hf auth login` or set `HF_TOKEN`. Inputs are 224 x 224 H&E tiles at 0.5 µm
  per pixel (one tile, a list or a directory); whole slides are not tiled.
- **MolE** `predict()` embeds the input table itself. To reuse an embedding,
  pass `model.embed(input)` as `predict(input, embedding)`; names and order
  must match the table.

## Adding models
If you want to add your own model to `vhmodels`, please follow our [contribution guideline](docs/model_contribution.md).

## Installing AI skills

Use your IDE's plugin or marketplace interface, or copy the contents of the
`skills` directory into the corresponding project directory (for example,
`.cursor/skills/` for Cursor).

For more details:

- [claude code](https://code.claude.com/docs/en/skills)
- [cursor](https://cursor.com/docs/skills)
- [codex](https://developers.openai.com/codex/skills)

## Folder structure

```text
vhmodels
├───.claude-plugin              # AI skills plugin metadata
├───docs                        # Manifest, contribution and Ray docs
├───example_data                # Example inputs per model
├───examples                    # Inference scripts; ray/ for the Ray runtime
├───hpc-job-submit              # Slurm scripts and config template
├───skills                      # AI agent skills
├───tests                       # Unit and integration tests
└───vhmodels
    ├───envs                    # Apptainer definition template
    ├───models                  # One folder per model + manifest registry
    │   ├───DinoBloom           # model.json, manifests/, model.py, deps
    │   ├───HOptimus0
    │   ├───Hyformer
    │   ├───MolE
    │   ├───Nicheformer
    │   ├───ProtTrans
    │   ├───registry.py         # Discovers and resolves manifests
    │   ├───schema.py           # Pydantic manifest schema
    │   ├───source_resolver.py  # Manifest sources to local paths
    │   └───discovery.py        # Dependency-free class lookup
    ├───utils                   # Shared helpers
    ├───vh_checker              # CLI, base model, Conda/Apptainer runtimes
    └───vh_ray                  # Ray runtime
```
