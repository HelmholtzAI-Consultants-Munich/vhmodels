# vhmodels on Kubernetes (PoC)

Ray on the Helmholtz Kubernetes cluster, namespace `chc`, **without KubeRay**:
the Ray head and workers are plain Deployments.

Models run on Ray workers started from their **OCI images on GHCR**, with the
vhmodels package exactly as the image ships it, so they use the same model
definitions as on Slurm: see [vhmodels from the GHCR images](#vhmodels-from-the-ghcr-images).
MolE is the first. The [legacy PoC](#legacy-poc-own-actor-code-per-model)
instead ran its own actor code per model and installed the dependencies at
start-up.

## Layout

```
k8s/
├── lib.sh                    # shared helpers for the test scripts
├── images/                   # vhmodels from the GHCR images (Ray Core and Serve)
│   ├── head.yaml             # Ray head: dashboard, Jobs API, Serve controller and proxy
│   ├── mole.yaml             # MolE worker node (StatefulSet), label vhmodels/project=mole
│   ├── serve.yaml            # Serve config (ConfigMap), deployed by the head on every start
│   └── run.sh                # end-to-end test
├── service/                  # legacy PoC: the inference service
│   ├── ray-head.yaml         # Ray head (dashboard, Jobs API, Ray Client)
│   ├── api.yaml              # HTTP API, POST /embed/<model>
│   └── models/
│       ├── hoptimus0/        # deploy.py (--gpu), ray-worker.yaml, ray-worker-gpu.yaml
│       └── prottrans/        # deploy.py, ray-worker.yaml
└── tests/                    # legacy PoC: each run.sh deploys the parts of service/ it needs
```

| Test | What it checks |
| --- | --- |
| [images/run.sh](images/run.sh) | MolE from its GHCR image: one Ray Core prediction and one Ray Serve HTTP prediction |
| [tests/smoke/](tests/smoke/) | Ray head + placeholder API, job submission via port-forward |
| [tests/hf-access/](tests/hf-access/) | A pod reaches Hugging Face over HTTPS with the token |
| [tests/hoptimus0/](tests/hoptimus0/) | H-Optimus-0 embeddings: API → Ray → model |
| [tests/two-models/](tests/two-models/) | H-Optimus-0 and ProtT5-XL-UniRef50 (11.3GB checkpoint) on two workers, under concurrent load |
| [tests/gpu/](tests/gpu/) | H-Optimus-0 on the GPU (fp16 autocast) vs the CPU actor: latency, embedding match, load |

Each `run.sh` (`smoke_test.sh` for the smoke test) creates its resources, runs
the check and **deletes what it created** afterwards. Set `KEEP=1` to leave it
running. Resources that were already running are reused and kept, so running a
test next to a kept service leaves that service intact. The scripts
port-forward to free local ports, so tunnels you have open on 8265 or 8000 are
not affected.

## vhmodels from the GHCR images

Images come from `.github/workflows/images.yaml`, tagged `ray-kubernetes`.
Every image holds `ray[serve]` and the vhmodels package, so every node runs the
same Python and the same vhmodels code:

| Pod | Image (`ghcr.io/helmholtzai-consultants-munich/…`) | Adds |
| --- | --- | --- |
| Head | `vhmodels-head` | nothing: no model dependencies |
| Model worker | `vhmodels-<model id>` | the model's dependencies |

```
Your Mac                          Namespace chc
────────                          ────────────────────────────────────────────────────

ray job submit ── port-forward ──►┌─────────────────────────────────────────────────┐
  examples/mole.py      8265      │ vhmodels-head (vhmodels-head image)             │
                                  │ job drivers, Serve controller and HTTP proxy    │
curl /mole/predict ─ port-fwd ───►│ serve deploy serve.yaml on every start          │
                                  │ no model: --num-cpus=0, no label                │
                        8000      └────────────────────────┬────────────────────────┘
                                                           │ label_selector
                                                           ▼ vhmodels/project=mole
                                  ┌─────────────────────────────────────────────────┐
                                  │ vhmodels-mole-0 (vhmodels-mole image)           │
                                  │ ray start --labels=vhmodels/project=mole        │
                                  │ ModelActor / Serve replica in /opt/venv:        │
                                  │ downloads the sources, load_model(), predict()  │
                                  │ volume cache-vhmodels-mole-0 (ssd) at /cache    │
                                  └────────────────────────┬────────────────────────┘
                                                           ▼ HTTPS (first run only)
                                                    huggingface.co
```

`VHMODELS_IMAGE_NODES=1`, set on every pod, is the only switch; scripts and
Serve configs are the same as on Slurm ([docs/ray.md](../docs/ray.md#kubernetes)):

| | Slurm | Kubernetes |
| --- | --- | --- |
| Model environment | venv or `.sif` (`VHMODELS_IMAGE_DIR`) | worker pods running the image (`VHMODELS_IMAGE_NODES=1`) |
| Placement | `runtime_env.py_executable` | `label_selector` `vhmodels/project=<id>` |
| Sources | the driver downloads them to the shared filesystem | the actor downloads them to its pod's volume |
| Ray Core inputs | shared filesystem | the job's `working_dir` |
| Serve inputs | paths on the shared filesystem | paths on the replica's pod |

**Run** ([images/run.sh](images/run.sh)): deploys the head, the MolE worker
and the Serve config, runs the unchanged `examples/mole.py` as a Ray job,
waits for `/mole` and POSTs the example table to `/mole/predict`.

**Serve apps survive head restarts.** A restarted head is an empty Ray
cluster; the workers rejoin by themselves, and the head's `postStart` hook runs
`serve deploy` on [images/serve.yaml](images/serve.yaml) (a ConfigMap) on every
start, so the apps come back too. Running Ray jobs are lost.

```bash
k8s/images/run.sh     # KEEP=1 leaves the cluster and the /mole app running
```

**Use a kept service:**

```bash
kubectl -n chc port-forward svc/vhmodels-head 8265:8265 8000:8000 &
.venv/bin/ray job submit --address http://localhost:8265 --working-dir <dir> -- python <script>.py
kubectl -n chc cp example_data/MolE/examples_molecules.tsv vhmodels-mole-0:/cache/in.tsv
curl -H 'Content-Type: application/json' -d '{"input": "/cache/in.tsv"}' localhost:8000/mole/predict
curl localhost:8000/mole/info

# change the Serve config: apply, wait until the mounted copy updates (up to a minute), deploy
kubectl apply -f k8s/images/serve.yaml
kubectl -n chc exec deploy/vhmodels-head -- serve deploy /etc/vhmodels/serve.yaml
```

**Cost per pod and request:**

- **Requests** run on a loaded model: each actor or replica loads it once.
  Replicas take one request at a time, so throughput grows with replicas.
  `OMP_NUM_THREADS` is set to the pod's CPUs; Ray would otherwise give each
  actor one thread.
- **Cold start** of a model pod: the image pull (about 5 min, once per node), the
  sources download (MolE 777MB, once per volume) and the model load.
- **Storage:** each worker pod of the StatefulSet has its own persistent
  volume (`ssd`, Ceph, one node at a time; 600Gi in the namespace), so the
  sources survive pod restarts and image updates. Deleting the StatefulSet
  keeps the volume: `kubectl -n chc delete pvc cache-vhmodels-mole-0`.
- **Images** carry two tags: `sha-<commit>`, fixed, and the branch name
  (`ray-kubernetes`), which moves with every push. The manifests follow the branch
  tag with `imagePullPolicy: Always`; pin a `sha-` tag once it is stable.

**Add a model:** copy `images/mole.yaml` with the model's image, name, label,
resources and volume size, and add its app to a Serve config. The head needs
no change: CI rebuilds its image in every run, with every model's manifest.
GPU workers request `nvidia.com/gpu: 1` and `--num-gpus=1`; a pod on the GPU
node can use about 14 CPUs and 56Gi.

### Moving to KubeRay

KubeRay is not installed in the cluster yet. When it is, these parts change;
everything else, the vhmodels package included, stays as it is.

| Now (plain Kubernetes) | With KubeRay |
| --- | --- |
| `head.yaml`: Deployment and Service, `ray start --head` | RayService `headGroupSpec`; the operator starts Ray and creates the head and Serve Services |
| `mole.yaml`: one StatefulSet per model, `ray start --labels=vhmodels/project=mole` | one `workerGroupSpecs` entry per model image, the same label in `rayStartParams` |
| `serve.yaml` ConfigMap, deployed by the head's `postStart` hook | `serveConfigV2` in the RayService; the operator deploys it, redeploys it after failures and upgrades without downtime |
| Fixed worker replicas | Autoscaling per worker group between `minReplicas` and `maxReplicas`, possibly to zero |
| Persistent volume per worker pod (StatefulSet) | Per-pod ephemeral volume, downloaded again for each new pod, unless shared storage exists |
| A head restart loses the cluster state; apps come back through the hook | The same, or none lost with GCS fault tolerance (external Redis) |
| `ray job submit` through a port-forward | The same, or a RayJob per job |
| Head runs the MolE image | Unchanged: KubeRay brings no image |
| `kubectl port-forward` for access | Unchanged: an Ingress needs DNS either way |

## Legacy PoC: own actor code per model

These parts predate the model images: each `deploy.py` defines its own actor
and installs torch and the model's packages when the worker starts.

### Architecture (H-Optimus-0 on CPU)

```
Your Mac                               Namespace chc
────────                               ──────────────────────────────────────────────

ray job submit ─┐                      ┌────────────────────────────────────────────┐
                ├─ port-forward 8265 ─►│ Ray head                                   │
browser ────────┘                      │ dashboard and Jobs API (8265)              │
                                       └──────▲──────────────────────────▲──────────┘
                                              │ Ray Client (10001)       │ joins (6379)
                                       ┌──────┴──────┐          ┌────────┴──────────┐
curl /embed ──── port-forward 8000 ───►│ API pod     │          │ Ray worker        │
                                       │ POST /embed │          │ H-Optimus-0 actor │
                                       └─────────────┘          └────────┬──────────┘
                                                                         │ HTTPS, HF_TOKEN from
                                                                         │ Secret hf-hopt0-token
                                                                         ▼
                                                                huggingface.co, PyPI
```

1. `deploy.py` is submitted as a **Ray job**. It starts a long-lived actor
   named `hoptimus0` on the worker. The actor installs CPU torch and timm,
   downloads the checkpoint (~4.5GB) and loads the model in fp32.
2. The **API pod** receives `POST /embed` with an image and calls that actor
   through the Ray head over Ray Client.
3. The actor returns the 1536-dim embedding.

## Resources

| Pod | Image | CPU | Memory | Runs |
| --- | --- | --- | --- | --- |
| `vhmodels-ray-head` | `rayproject/ray:2.59.0-py311` | 2 | 4Gi | Ray coordinator, dashboard, job API, job drivers |
| `vhmodels-head` | `ghcr.io/helmholtzai-consultants-munich/vhmodels-head:ray-kubernetes` | 1-2 | 4Gi | Ray head of `images/`: job drivers, Serve controller and proxy |
| `vhmodels-mole-0` | `ghcr.io/helmholtzai-consultants-munich/vhmodels-mole:ray-kubernetes` | 2 | 6Gi + 5Gi volume | MolE actor and Serve replica (CPU) |
| `vhmodels-ray-worker` | `rayproject/ray:2.59.0-py311` | 2 | 8Gi | H-Optimus-0 actor (CPU, fp32) |
| `vhmodels-ray-worker-prottrans` | same | 2 | 8Gi | ProtT5-XL-UniRef50 encoder actor (CPU, fp32) |
| `vhmodels-ray-worker-gpu` | `rayproject/ray:2.59.0-py311-cu126` | 2 + 1 GPU | 8Gi | H-Optimus-0 actor `hoptimus0-gpu` (CUDA, fp16 autocast) |
| `vhmodels-inference-api` | same | 0.1 | 0.5Gi | HTTP API on port 8000 |
| `vhmodels-api-smoke` (smoke test) | same | 0.1 | 0.5Gi | placeholder API |
| `vhmodels-hf-access` (hf-access test) | `python:3.11-slim` | 0.1 | 128Mi | one-off HF check |

- Namespace quota: 16 CPU, 64Gi, 1 GPU.
- Our pods have only been scheduled on two nodes, `hpc-k8s-md-0-vf4q7-8hkmp`
  and `hpc-k8s-md-0-vf4q7-gbtvq`. Their capacity is not visible to us (we
  cannot list nodes); one has taken 10.5Gi of our requests at once.
- Both nodes have **AMD EPYC Milan** CPUs (AVX2, no AVX-512).
- The GPU pod runs on a separate GPU node, `hpc-k8s-md-gpu-kl2fj-bvh8m`, with
  an **NVIDIA A100 80GB PCIe** (driver 595.91, CUDA 13.2). H-Optimus-0 there
  takes ~0.4 s per tile against ~6 s on 2 CPUs, with the same embeddings
  (cosine similarity ≥ 0.99999).

## What is stored where

| What | Where | Lifetime |
| --- | --- | --- |
| HF token | Secret `hf-hopt0-token` in `chc` | **Permanent**, until deleted |
| Manifests, scripts | this repo (`k8s/`) | git |
| Model images (GHCR) | container image cache of each node | Until the node evicts them |
| MolE sources, Serve inputs | volume `cache-vhmodels-mole-0` (`ssd`) at `/cache` | **Permanent**, until the PVC is deleted |
| Pods, Services, ConfigMaps | `chc` | Until the script deletes them |
| CPU torch, timm (pip) | worker container, `/tmp/ray` | Gone with the pod |
| Model checkpoint | worker `emptyDir` `/cache/huggingface` | Gone with the pod |
| Job logs, uploaded job code | Ray head, `/tmp/ray` | Gone with the pod |

No persistent disk is used. Each new worker reinstalls torch and re-downloads
the model (~3 min).

## Setup (once)

```bash
brew install kubectl int128/kubelogin/kubelogin
export KUBECONFIG=~/Documents/kubeconfig-chc.yaml   # in ~/.zshrc
kubectl auth whoami                                  # browser login

# vhmodels Ray driver venv, as in the main README (Python 3.11, ray[default]==2.59.0)
uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[cli,ray]"
```

This is the venv from [Ray runtime](../README.md#ray-runtime-experimental) in
the main README; its Ray matches the cluster. The scripts use `.venv/bin/ray`
and stop with a hint if it is missing, is not version 2.59.0 or lacks the
`ray[default]` extras that `ray job` needs. Use `.venv/bin/ray` (or
`source .venv/bin/activate`) for your own `ray` commands too: a `ray` from
another Python on your `PATH` may be a different version or a plain `ray`
install. To use a Ray CLI elsewhere, set `RAY=/path/to/ray`.

The token Secret already exists. To recreate it:

```bash
read -s HF_TOKEN && kubectl -n chc create secret generic hf-hopt0-token --from-literal=HF_TOKEN="$HF_TOKEN"; unset HF_TOKEN
```

## How to use

**Run the tests** (from the repo root):

```bash
k8s/tests/smoke/smoke_test.sh
k8s/tests/hf-access/run.sh
k8s/tests/hoptimus0/run.sh     # add KEEP=1 to keep the service running
k8s/tests/two-models/run.sh
k8s/tests/gpu/run.sh
```

**Keep the inference service running and use it:**

```bash
KEEP=1 k8s/tests/hoptimus0/run.sh

# in a separate terminal, open the tunnels
kubectl -n chc port-forward svc/vhmodels-ray-head 8265:8265 &
kubectl -n chc port-forward svc/vhmodels-inference-api 8000:8000 &

# request an embedding (224 x 224 RGB tile)
curl --data-binary @example_data/H-Optimus-0/TUM-AACHEHDV.tif http://localhost:8000/embed
```

With `KEEP=1 k8s/tests/two-models/run.sh` instead, ProtT5 embeds protein sequences
(one mean-pooled 1024-dim embedding per sequence, at most 4000 residues per
request):

```bash
curl -H 'Content-Type: application/json' -d '{"sequences": ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"]}' \
    http://localhost:8000/embed/prottrans
```

With `KEEP=1 k8s/tests/gpu/run.sh`, the GPU actor answers on
`/embed/hoptimus0-gpu`.

**Ray dashboard:** with the 8265 port-forward open, browse to
<http://localhost:8265>. It shows jobs, logs, actors and resource use.

**Job output (`print`)**:

```bash
.venv/bin/ray job list --address http://localhost:8265
.venv/bin/ray job logs --address http://localhost:8265 <job-id>    # -f to follow
```

**Watch the pods:**

```bash
kubectl -n chc get pods -w
kubectl -n chc logs -f deploy/vhmodels-ray-worker
```

**Stop everything** (frees the resources):

```bash
kubectl -n chc delete -R -f k8s/service -f k8s/tests/smoke/api.yaml --ignore-not-found
```

## Security

- Ray head ports (8265, 10001) have **no authentication**. They stay internal
  (`ClusterIP`) and are reached only through `kubectl port-forward`.
- Only the inference API will get a public hostname
  (`vhmodels.helmholtz-munich.de`, campus/VPN), once DNS and TLS are ready.

## Next steps

1. **Inputs over HTTP.** Serve takes file paths, which must exist on the
   replica's pod; MolE's table is copied there first.
2. More models, then GPU workers on the GPU node.
3. Ingress with HTTPS for the Serve proxy.
