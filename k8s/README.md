# vhmodels on Kubernetes (PoC)

Ray on the Helmholtz Kubernetes cluster, namespace `chc`, **without KubeRay**:
the Ray head and workers are plain Deployments. The models run on **CPU**,
and H-Optimus-0 also on the namespace's one GPU.

## Layout

```
k8s/
├── lib.sh                    # shared helpers for the test scripts
├── service/                  # the inference service: what gets deployed
│   ├── ray-head.yaml         # Ray head (dashboard, Jobs API, Ray Client)
│   ├── api.yaml              # HTTP API, POST /embed/<model>
│   └── models/
│       ├── hoptimus0/        # deploy.py (--gpu), ray-worker.yaml, ray-worker-gpu.yaml
│       └── prottrans/        # deploy.py, ray-worker.yaml
└── tests/                    # each run.sh deploys the parts of service/ it needs
```

There is one service, and the tests deploy different parts of it; they do not
build on each other, and each runs on its own.

| Test | What it checks |
| --- | --- |
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

## Architecture (H-Optimus-0 on CPU)

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
| `vhmodels-ray-worker` | same | 2 | 8Gi | H-Optimus-0 actor (CPU, fp32) |
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

- Custom image with the model dependencies, pushed to a registry.
- Persistent volume for the HF cache.
- Ingress with HTTPS for the API.
