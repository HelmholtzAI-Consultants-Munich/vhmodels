# Ray runtime (experimental)

`runtime="ray"` runs each model as a [Ray](https://docs.ray.io) actor in the
model's own environment: its Apptainer image, its uv venv or, on Kubernetes,
its image's worker pods. Scripts are the same as for the Conda and Apptainer
runtimes, which remain the default.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

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

- **Cluster.** `load_model` connects to the cluster in `RAY_ADDRESS`, otherwise it starts a local one.
- **GPUs.** `device="cuda:0"` gives the actor one GPU. Other actor options go in `ray_options`, e.g. `{"num_gpus": 0.5}`.
- **Examples.** The scripts in [examples/](../examples/) run unchanged with `VHMODELS_RUNTIME=ray`. [examples/ray/two_models.py](../examples/ray/two_models.py) runs two models in parallel.
- **Inputs.** Paths must be visible where the actor runs; relative paths need a shared filesystem.

## Model environment

| Setting | The actor runs in |
| --- | --- |
| `load_model(..., image_path=...)` | that `.sif` image |
| `VHMODELS_IMAGE_DIR` | `<dir>/vhmodels-<project>.sif`, pulled from GHCR ([images.md](images.md)) or built locally |
| neither | the model's venv under `VHMODELS_VENV_DIR` (default `~/.cache/vhmodels/venvs`) |
| `VHMODELS_IMAGE_NODES=1` | the model's worker pods on [Kubernetes](#kubernetes) |

- **Venvs.** `vh-checker create-ray-env <project>` (or `--all`) builds the venv, and rebuilds it only when something changed; `--force` always rebuilds. `load_model` never builds: it raises if the venv is missing or outdated.
- **Images** start faster than venvs on a network filesystem such as Lustre, where imports can take minutes.
- **Versions.** Images and venvs have the driver's Python (3.11.15) and Ray (2.59.0). Use the `.venv` above as the driver.
- **On a cluster,** put `VHMODELS_VENV_DIR` and `UV_CACHE_DIR` on shared storage.
- **Apptainer mounts.** Inputs, outputs and caches outside Apptainer's default mounts go in `APPTAINER_BINDPATH`, set before Ray starts.

## Slurm cluster

[`hpc-job-submit/ray_cluster.sbatch`](../hpc-job-submit/ray_cluster.sbatch)
starts a long-lived single-node Ray cluster; scripts run on it as
[Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html).

| Step | Command (login node) |
| --- | --- |
| Start | `sbatch hpc-job-submit/ray_cluster.sbatch /abs/config.json [/abs/serve.yaml]`; the log prints `RAY_ADDRESS`; a Serve config is deployed at start |
| Connect | `export PATH=<repo>/.venv/bin:$PATH RAY_AUTH_MODE=token RAY_ADDRESS=<from the log>` |
| Submit | `ray job submit --working-dir examples -- python mole.py` |
| Monitor | `ray job list`, `ray job logs -f <id>`, dashboard at `RAY_ADDRESS` |
| Cancel a job | `ray job stop <id>` |
| Serve models | `serve deploy examples/ray/serve.yaml`; see [Ray Serve](#ray-serve) |
| Stop the cluster | `scancel <cluster job id>` (default time limit: 30 min) |

- **Scripts are unchanged.** `load_model(runtime="ray")` inside a job connects to the cluster; `--no-wait` returns right after submitting.
- **Environment.** Jobs get the caches, `VHMODELS_RUNTIME=ray`, `VHMODELS_DEVICE=cuda:0`, `VHMODELS_IMAGE_DIR`, `VHMODELS_DATA_DIR` and `VHMODELS_OUTPUT_DIR` from the config. Per-job settings go in `--runtime-env-json`.
- **Paths.** Inputs and outputs must live under `DATA_DIR` or `OUTPUT_DIR`; model images see only those and the caches.
- **GPUs.** 2 GPUs by default, one per model actor. Other GPUs, e.g. A100 MIG slices: `sbatch --gres=gpu:a100_3g.20gb:2 ...` (longer queue).
- **Token.** The cluster accepts only clients that send its token (`~/.ray/auth_token`); `RAY_AUTH_MODE=token` in Connect does that.

## Ray Serve

Ray Serve keeps models loaded as HTTP apps, one per model, until you remove
them or the cluster ends; failed replicas are restarted. With Ray Core
(`load_model`), a model lives only as long as your script.

| Step | Command (login node, connected as above) |
| --- | --- |
| Connect | As above, plus `export RAY_DASHBOARD_ADDRESS=$RAY_ADDRESS`; the `serve` CLI ignores `RAY_ADDRESS` |
| Deploy | `serve deploy examples/ray/serve.yaml` (one app per model), or pass the config to `ray_cluster.sbatch` |
| Status | `serve status`; the dashboard's Serve tab |
| Embed | `curl -X POST <serve URL>/hyformer/embed -H 'Content-Type: application/json' -d '{"input": [...]}'` |
| Upload files | `curl -F input=@examples_molecules.tsv <serve URL>/mole/predict/files`; see [File uploads](#file-uploads) |
| Other routes | `POST <app>/predict`, `GET <app>/info`; the body holds the keyword arguments of `embed`/`predict` |
| API docs | `<serve URL>/<app>/docs` (OpenAPI) |
| Remove | `serve delete <app>` or `serve shutdown` |

- **Serve URL.** `ray_cluster.sbatch` prints it.
- **Config.** In an app's `args`, `ray_options` goes to the replica (e.g. `num_gpus: 0.5`) and every other key to `load_model`. `num_replicas` or `autoscaling_config` go under the app's `deployments` (name `ModelDeployment`); do not set `ray_actor_options` there, it replaces the model's environment.
- **One request at a time per replica;** further requests wait in a queue.
- **Input** has the shape of the model's `implementation.input` ([manifest.md](manifest.md#input)), which `GET <app>/info` shows. Other body fields go to the model as keyword arguments. Paths resolve on the replica, so use absolute paths under `DATA_DIR`, or upload the files.
- **Responses** are the model's result, e.g. `{"output": [[...]]}`.
- **Python clients** on the cluster can skip HTTP: `serve.get_app_handle("hyformer").embed.remote(input=[...]).result()`.

| Status | Cause |
| --- | --- |
| 422 | Body is not JSON, or `input` does not match the model's `implementation.input`; for uploads, no `input` file or `kwargs` not a JSON object |
| 415 | Upload to a model that takes JSON input (`sequences`) |
| 400 | The model raised `ValueError`, `TypeError` or `FileNotFoundError` (bad input or kwargs) |
| 501 | The model raised `NotImplementedError` (e.g. `predict` on an embed-only model) |
| 500 | Any other model error |

### File uploads

Clients that don't share the replica's disk (laptops, web apps) POST the
files to `<app>/embed/files` or `<app>/predict/files` as `multipart/form-data`.
The replica keeps them only for that request.

| `implementation.input` | Form fields | Example |
| --- | --- | --- |
| `file` | one `input` file | `curl -F input=@examples_molecules.tsv <serve URL>/mole/predict/files` |
| `images` | one or more `input` files, embedded in upload order | `curl -F input=@a.tif -F input=@b.tif <serve URL>/hoptimus0/embed/files` |
| `named_files` | one file per key | `curl -F technology_mean=@mean.npy -F data=@cells.h5ad <serve URL>/nicheformer/embed/files` |

- **Keyword arguments** go in a `kwargs` field as a JSON object, e.g. `-F 'kwargs={"batch_size": 8}'`.
- **Large files** on the cluster: paths under `DATA_DIR` avoid the upload.

## Kubernetes

With `VHMODELS_IMAGE_NODES=1`, each model runs on Ray worker pods started from
its image, and the same scripts and Serve configs run as on Slurm. Input paths
must exist on the pod: pass the files in a Ray job's `--working-dir`, or upload
them to Serve. Setup and usage: [k8s/README.md](../k8s/README.md).
