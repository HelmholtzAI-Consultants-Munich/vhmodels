# Ray runtime (experimental)

`runtime="ray"` runs each model as a [Ray](https://docs.ray.io) actor in the model's own Apptainer image or uv venv. The Conda and Apptainer runtimes are unchanged and remain the default.

```text
driver (Python 3.11.15, vhmodels[ray]: ray[default]==2.59.0)
  vhmodels.load_model(project, model, runtime="ray", device=...)
    ├─ Registry.resolve_resources()   manifest + downloads, on the driver
    └─ RayModel                       vhmodels/vh_ray/client.py
         ray.remote(ModelActor).options(runtime_env={"py_executable": <image or venv python>})
           └─ ModelActor              vhmodels/vh_ray/actor.py
                └─ model.load_model(variant, resources) / embed / predict
```

- Ray starts the actors, places them on CPUs and GPUs, serializes requests and results, and returns model errors; there is no socket worker or process manager.
- `ModelActor` is a plain class made an actor with `ray.remote(...)`, so it can also back a Ray Serve deployment later.

## Model venvs

`vh-checker create-ray-env <project>` builds `<VENV_DIR>/<project>` from the same
uv spec that the Apptainer image uses: `requirements`, `exclude` and
`torch_backend` under `runtimes.uv` in `model.json`. It then adds two things:

- `ray==<driver Ray version>`, resolved in the same `uv pip install` as the model's pins, so conflicts fail at build time rather than when the actor starts.
- An editable, dependency-free install of this checkout, so the actor can import `vhmodels`. This is the counterpart of `PYTHONPATH=/opt/vhmodels-src` in the image.

Bytecode is compiled at build time (`--compile-bytecode`). The build ends with
a smoke import of the model module, run with `python -I` so that it imports
from the venv rather than the current directory.

Ray runs every actor with the driver's worker entrypoint and only swaps the
interpreter. That is why a model venv must match the driver exactly:

- **Python**: the venv is built with the driver's exact `platform.python_version()` as a uv-managed CPython. Model environments do not choose their Python: `MODEL_PYTHON` in `vhmodels/utils/uv_utils.py` (**3.11.15**, matching `.python-version`) applies to every Apptainer image and Ray venv. If the driver runs another version, the build warns and still uses the driver's.
- **Ray**: the version is pinned once, in the `ray` extra of `pyproject.toml`.

### When a venv is rebuilt

The build writes `vhmodels-ray-env.json` into the venv as its last step, so the file exists only after a complete build. It records:

- the Python and Ray versions;
- the SHA-256 of the requirements and exclude files;
- the exact uv build commands, which include the torch backend, build flags and the vhmodels checkout path.

`create-ray-env` (and `build_ray_env.sbatch`) compares this record with what the current driver and manifest would produce:

- **Equal:** the existing venv is reused and nothing is installed.
- **Different, or no marker:** the venv is deleted and rebuilt from scratch. Only a directory containing the marker or a `pyvenv.cfg` is ever deleted.
- **`--force`:** always rebuilds.

`load_model(runtime="ray")` never builds. It compares the same full record and raises, naming what changed, if the venv is missing or outdated (for example after editing a requirements file).

| Setting | Default |
| --- | --- |
| `VHMODELS_VENV_DIR` | `${XDG_CACHE_HOME:-~/.cache}/vhmodels/venvs` |
| `UV_CACHE_DIR` | `${XDG_CACHE_HOME:-~/.cache}/vhmodels/apptainer/uv`, which Apptainer builds also use |

On a cluster, put both on shared storage. The same `py_executable` path then works on every node.

## Model images

The actor runs in the model's Apptainer image instead of its venv when an image is given:

| Source | Image |
| --- | --- |
| `load_model(..., image_path=...)` | that file |
| `VHMODELS_IMAGE_DIR` | `<dir>/vhmodels-<project>.sif`; must exist |
| neither | the venv under `VHMODELS_VENV_DIR` |

- **Launch.** `py_executable` is `apptainer exec [--nv] <image> /opt/venv/bin/python`; `--nv` when the actor requests a GPU. Imports come from the image's squashfs, not the network filesystem.
- **Versions.** Every image installs Python `MODEL_PYTHON` and `ray==RAY_VERSION` (`vhmodels/utils/uv_utils.py`), which must equal the driver's exactly.
- **Code.** The actor imports the vhmodels code baked into the image.
- **Bind mounts.** Ray's session dir (`/tmp/ray`) is mounted by default. The driver env (Ray's worker entrypoint), caches, inputs and outputs go in `APPTAINER_BINDPATH`, set before Ray starts; workers inherit the raylet's environment. `ray_cluster.sbatch` sets both.

## Quick start

Install [uv](https://docs.astral.sh/uv/), for example the version the image build uses:
`curl -LsSf https://astral.sh/uv/0.12.3/install.sh | sh`. Then:

```bash
uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[cli,ray]"
.venv/bin/vh-checker create-ray-env hyformer
```

```python
import vhmodels

with vhmodels.load_model(
    project="hyformer", model="hyformer_molecules_8M", runtime="ray"
) as model:
    print(model.info())  # waits for the load; shows the actor's interpreter
    embeddings = model.embed(input=["O=C(c1cccc(/N=C(\\O)CCc2ccccc2)c1)[N+]1CCCCC1"])
```

- **Ray connection.** `load_model` connects to the cluster in `RAY_ADDRESS`, otherwise it starts a local one. Call `ray.init(...)` yourself first to control this.
- **GPUs.** A `device` starting with `cuda` requests one GPU for the actor. Ray then sets the actor's `CUDA_VISIBLE_DEVICES`, so `cuda:0` is the assigned GPU.
- **Actor options.** `ray_options={...}` is passed to `.options()`, for example `{"num_gpus": 0.5}`.
- **Parallel calls.** `RayModel.actor` is the actor handle. Calls such as `model.actor.embed.remote(...)` return futures; [examples/ray/two_models.py](../examples/ray/two_models.py) uses them to run two models with conflicting dependencies concurrently.
- **Same API as the other runtimes.** The scripts in [examples/](../examples/) run unchanged with `VHMODELS_RUNTIME=ray`.
- **Load time.** The actor starts loading when `load_model` returns; the first call waits for it. Calls have no timeout by default; `RayModel(..., timeout=...)` sets one.
- **Input paths.** Relative input paths work because each request carries the caller's working directory (needs a shared filesystem).

### Start-up cost on network filesystems

The venvs live on a shared filesystem, so imports pay its metadata latency. Apptainer avoids this because a SIF image is a local squashfs.

On the HPC Lustre, importing Hyformer's tokenizer stack (`deepchem`) takes about 15 s warm and about 1 min cold on a CPU node. A cold actor start, which also imports torch and loads the checkpoint, took several minutes before bytecode was precompiled.

`create-ray-env` therefore compiles bytecode at build time. Running the actor in the model's image (see above) removes the rest of this cost.

### Driver paths on worker `sys.path`

- Ray prepends the driver's script directory and working directory (a job's `working_dir`) to every worker's `sys.path`, so a script named like a model dependency shadows it. Example: `examples/hyformer.py` shadowed the `hyformer` package and deadlocked the actor.
- `RayModel` passes these directories to `ModelActor`, which removes them from `sys.path` before loading the model.

### Authentication

- Ray 2.59 protects every cluster with a token in `~/.ray/auth_token`.
- Clients send it only with `RAY_AUTH_MODE=token`; `ray_cluster.sbatch` sets it for the cluster, set it yourself on the login node.

## Slurm cluster

[`hpc-job-submit/ray_cluster.sbatch`](../hpc-job-submit/ray_cluster.sbatch) starts a long-lived single-node cluster with `ray start --head --block`, following the multi-user setup in [Ray's Slurm guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html); scripts run on it as [Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html).

| Step | Command (login node) |
| --- | --- |
| Start | `sbatch hpc-job-submit/ray_cluster.sbatch /abs/config.json`; the log prints `RAY_ADDRESS` |
| Connect | `export PATH=<repo>/.venv/bin:$PATH RAY_AUTH_MODE=token RAY_ADDRESS=<from the log>` |
| Submit | `ray job submit --working-dir examples -- python mole.py` |
| Monitor | `ray job list`, `ray job logs -f <id>`, dashboard at `RAY_ADDRESS` |
| Cancel a job | `ray job stop <id>` |
| Stop the cluster | `scancel <cluster job id>` (default time limit: 30 min) |

- **Scripts are unchanged.** `load_model(runtime="ray")` inside a job connects to the cluster; `--no-wait` returns right after submitting.
- **Inherited environment.** Jobs and actors get the caches, `VHMODELS_RUNTIME=ray`, `VHMODELS_DEVICE=cuda:0`, `VHMODELS_IMAGE_DIR`, `VHMODELS_DATA_DIR` and `VHMODELS_OUTPUT_DIR` from the config. Per-job settings go in `--runtime-env-json`.
- **Paths.** Model images see only `DATA_DIR`, `OUTPUT_DIR` and the caches, so inputs and outputs must live there.
- **GPUs.** Default: 2 GPUs of any type the model images support, via `--constraint`; Ray counts `GPU: 2`. Each model actor takes one GPU; the head and job drivers use CPUs only. Other GPUs: e.g. `sbatch --gres=gpu:a100_3g.20gb:2 ...` for A100 MIG slices (longer queue).
- **No B300.** The model images' PyTorch builds have no kernels for B300 (`no kernel image is available`), so the constraint leaves those nodes out.
- **Shared node.** Whole nodes wait hours, so the job shares its node. Ports and the temp dir (`/tmp/ray-<job id>`) come from the job id, as Ray's guide requires for several Ray clusters on one node. Only the dashboard port must be reachable from the login node.

| Firewall (measured 2026-10-22) | Open TCP ports |
| --- | --- |
| Login node → compute node | 8000–9000, 60001–64000 |
| Compute node ↔ compute node | 60001–64000 |
- **Single node.** Multi-node needs workers joining with `ray start --address=<head> --block`; see the roadmap.

## Checks

- `VHMODELS_RUN_RAY_INTEGRATION=1 pytest tests/test_ray_integration.py` builds a minimal venv and drives the dependency-free `PersistentFake` model through `load_model(runtime="ray")`.

## Roadmap

1. **Multi-node Slurm.** Let GPU jobs join the cluster as workers (`ray start --address=<head> --block`), with the head on a CPU node, and add GPU-type resources. Every Ray port must then be in 60001–64000.
2. **Registry integration.** Add an optional `resources` block to `model.json` next to `runtimes.uv` (read only by `utils.uv_utils.model_uv_spec`). Add `create-ray-env --all`, then make Ray the default runtime and retire Conda and Apptainer.
3. **Ray Serve.** `ModelActor` loads its model in `__init__`, so it can back a deployment directly. Pass the same `runtime_env` in `ray_actor_options`, add `ray[serve]` to the model venvs, and put a FastAPI ingress in front.
4. **Kubernetes (KubeRay).** Bake the driver env and the model venvs into one image at a fixed `VHMODELS_VENV_DIR`. The `py_executable` mechanism stays the same.
