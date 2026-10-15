# Ray runtime (experimental)

The Ray runtime runs each model as a [Ray](https://docs.ray.io) actor inside the
model's own uv venv. Ray replaces the Unix-socket worker and process manager
used by the Conda and Apptainer runtimes. It starts actors, places them on CPUs
and GPUs, serializes requests and results, and passes model errors back to the
caller. The Conda and Apptainer runtimes are unchanged and remain the default.

```text
driver (Python 3.11, vhmodels[ray]: ray[default]==2.59.0)
  vhmodels.load_model(project, model, runtime="ray", device=...)
    ├─ Registry.resolve_resources()   manifest + downloads, on the driver
    └─ RayModel                       vhmodels/vh_ray/client.py
         ray.remote(ModelActor).options(runtime_env={"py_executable": <VENV_DIR>/<project>/bin/python})
           └─ ModelActor              vhmodels/vh_ray/actor.py
                └─ BaseModel.get_class(project)().load_model(variant, resources) / embed / predict
```

`ModelActor` is an ordinary Ray actor class. It is a plain Python class, and
`ray.remote(ModelActor)` turns it into an actor, the standard Ray pattern.
Leaving the decorator off keeps the class usable as a Ray Serve deployment
later. The class loads the model in `__init__` and calls the model's methods
directly; no socket, message protocol or worker process manager is involved.

## Model venvs

`vh-checker create-ray-env <project>` builds `<VENV_DIR>/<project>` from the same
uv spec that the Apptainer image uses: `requirements`, `exclude` and
`torch_backend` under `runtimes.apptainer` in `model.json`. It then adds two
things:

- `ray==<driver Ray version>`, resolved in the same `uv pip install` as the model's pins, so conflicts fail at build time rather than when the actor starts.
- An editable, dependency-free install of this checkout, so the actor can import `vhmodels`. This is the counterpart of `PYTHONPATH=/opt/vhmodels-src` in the image.

Bytecode is compiled at build time (`--compile-bytecode`). The build ends with
a smoke import of the model module, run with `python -I` so that it imports
from the venv rather than the current directory.

Ray runs every actor with the driver's worker entrypoint and only swaps the
interpreter. That is why a model venv must match the driver exactly:

- **Python**: the venv is built with the driver's exact `platform.python_version()` as a uv-managed CPython. The cluster standardises on **Python 3.11**. If a manifest declares another version, the build warns and uses the driver's version.
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

Unlike the other runtimes, the actor is created and starts loading as soon as `load_model` returns.

- **Load time.** The first call waits for the load. Calls have no timeout by default; `RayModel(..., timeout=...)` sets one per call.
- **Input paths.** Relative input paths still work because each request carries the caller's working directory; that requires a shared filesystem.

### Start-up cost on network filesystems

The venvs live on a shared filesystem, so imports pay its metadata latency. Apptainer avoids this because a SIF image is a local squashfs.

On the HPC Lustre, importing Hyformer's tokenizer stack (`deepchem`) takes about 15 s warm and about 1 min cold on a CPU node. A cold actor start, which also imports torch and loads the checkpoint, took several minutes before bytecode was precompiled.

`create-ray-env` therefore compiles bytecode at build time. Venvs that live on node-local disk or in an image remove the rest of this cost; see the roadmap below.

### Driver paths on worker `sys.path`

Unless a job sets a runtime_env `working_dir`, Ray prepends the driver's script directory and working directory to every worker's `sys.path`, ahead of site-packages. A driver script named like a model dependency is then imported in its place. For example, `examples/hyformer.py` shadowed the `hyformer` package: the actor re-ran the example and deadlocked waiting on a nested actor.

`RayModel` therefore passes these directories to `ModelActor`, which removes them from `sys.path` before loading the model. The actor class itself is imported earlier, so a driver directory containing another `vhmodels` checkout would still shadow the installed one.

### Authentication

Ray 2.59 protects every cluster it starts with a token, stored in `~/.ray/auth_token`. A driver that *connects* to a running cluster sends the token only when `RAY_AUTH_MODE=token` is set.

`run_inference_ray.sbatch` sets this for the whole job. Keep authentication on: other users on shared nodes can reach the cluster's ports.

## Slurm

See [hpc-job-submit/README.md](../hpc-job-submit/README.md):

- `build_ray_env.sbatch` builds the venv of the configured model.
- `run_inference_ray.sbatch` starts a single-node Ray head inside a GPU allocation and runs the inference script against it.

Notes on `run_inference_ray.sbatch`:

- **Ports.** The GCS, Ray Client and dashboard-agent ports have fixed defaults, so the script derives them from the job id. All other ports are random.
- **Temp dir.** Ray creates Unix sockets under `<temp-dir>/session_<timestamp>/sockets/`, and those paths must stay under 108 bytes. That leaves about 40 bytes for the temp dir, so the script falls back from `/localscratch/$USER/ray-<job>` to `/tmp/ray-<job>` when needed.
- **Shutdown.** Slurm stops the cluster when the job ends.

## Checks

- `VHMODELS_RUN_RAY_INTEGRATION=1 pytest tests/test_ray_integration.py` builds a minimal venv and drives the dependency-free `PersistentFake` model through `load_model(runtime="ray")`.

## Roadmap

1. **Multi-node Slurm.** Use `srun ray symmetric-run` (Ray ≥ 2.49), and add GPU-type resources. Cut the network-filesystem start-up cost, for example by packing each venv into a squashfs image or copying it to node-local disk at job start.
2. **Registry integration.** Add optional `runtimes.uv` and `resources` blocks to `model.json`. `utils.uv_utils.model_uv_spec` is the single place that reads them. Add `create-ray-env --all`, move the remaining models to Python 3.11, then make Ray the default runtime and retire Conda and Apptainer.
3. **Ray Serve.** `ModelActor` loads its model in `__init__`, so it can back a deployment directly. Pass the same `runtime_env` in `ray_actor_options`, add `ray[serve]` to the model venvs, and put a FastAPI ingress in front.
4. **Kubernetes (KubeRay).** Bake the driver env and the model venvs into one image at a fixed `VHMODELS_VENV_DIR`. The `py_executable` mechanism stays the same.
