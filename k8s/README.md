# vhmodels on Kubernetes (experimental)

Runs the [Ray runtime](../docs/ray.md) on Kubernetes, in namespace `chc`: a
Ray head pod and, per model, a Ray worker pod started from the model's
[image](../docs/images.md). Scripts and Serve configs are the same as on
Slurm. MolE is the first model.

The Ray and Serve ports have no authentication; reach them only through
`kubectl port-forward`.

```text
k8s/
├── head.yaml     Ray head: dashboard, Ray Jobs, Ray Serve
├── mole.yaml     MolE worker
├── serve.yaml    Serve config, deployed by the head on every start
└── run.sh        end-to-end test
```

## Setup (once)

```bash
brew install kubectl int128/kubelogin/kubelogin
export KUBECONFIG=<path to the namespace's kubeconfig>
kubectl auth whoami

uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[cli,ray]"
```

Put the `export` in `~/.zshrc`; `kubectl auth whoami` opens the browser login.
The `.venv` has the cluster's Ray version (2.59.0); use `.venv/bin/ray` for
`ray` commands.

## How to use

**Run the test** from the repo root. [run.sh](run.sh) starts the head, the
MolE worker and the Serve config, runs `examples/mole.py` as a Ray job and
uploads the example table to `/mole/predict/files`. Afterwards it deletes what
it started; `KEEP=1` leaves it running. The first start takes several minutes
(image pull and model download).

```bash
k8s/run.sh
KEEP=1 k8s/run.sh
```

**Use a running service.** Open the tunnels in one terminal and leave it open:

```bash
kubectl -n chc port-forward svc/vhmodels-head 8265:8265 8000:8000
```

Then, in another terminal:

```bash
curl -F input=@example_data/MolE/examples_molecules.tsv localhost:8000/mole/predict/files
curl localhost:8000/mole/info
.venv/bin/ray job submit --address http://localhost:8265 --working-dir <dir> -- python <script>.py
.venv/bin/ray job logs -f --address http://localhost:8265 <job id>
```

The dashboard is at <http://localhost:8265>: jobs, logs, actors and Serve apps.
See [Ray Serve](../docs/ray.md#ray-serve) for the other routes.

**Change the Serve config:** apply it, wait up to a minute until the head sees
the change, then deploy it:

```bash
kubectl -n chc apply -f k8s/serve.yaml
kubectl -n chc exec deploy/vhmodels-head -- serve deploy /etc/vhmodels/serve.yaml
```

**Watch the pods:**

```bash
kubectl -n chc get pods -w
kubectl -n chc logs -f vhmodels-mole-0
```

**Stop everything.** The second command also deletes MolE's downloaded files:

```bash
kubectl -n chc delete -f k8s/serve.yaml -f k8s/head.yaml -f k8s/mole.yaml
kubectl -n chc delete pvc cache-vhmodels-mole-0
```

## Add a model

Copy `mole.yaml` with the model's image, name, label
(`vhmodels/project=<model id>`), resources and volume size, and add its app to
`serve.yaml`.

- **GPU workers** request `nvidia.com/gpu: 1` and `--num-gpus=1`.
- **Gated models** (H-Optimus-0) need `HF_TOKEN` in the worker's `env`, from a
  Kubernetes Secret (`valueFrom.secretKeyRef`).
