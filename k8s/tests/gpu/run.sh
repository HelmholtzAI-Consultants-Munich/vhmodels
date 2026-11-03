#!/usr/bin/env bash
# H-Optimus-0 on the namespace's GPU, next to the CPU actor, on the chc namespace:
#   1. deploy the Ray head, the CPU model worker, a GPU worker and the API,
#   2. print nvidia-smi from the GPU worker,
#   3. submit deploy.py and deploy.py --gpu, which start both actors,
#   4. compare.py: CPU vs GPU latency and embedding similarity per tile,
#   5. load.py against the GPU actor alone,
#   6. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running are reused and kept.
#
# DURATION (default 60s) and CONCURRENCY (default 4) set the load phase. Needs
# the hf-hopt0-token Secret and the repo's .venv with ray[default]==2.59.0
# (k8s/README.md, Setup); override with RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/k8s/lib.sh"

cleanup() {
    stop_port_forwards
    if [[ "${KEEP:-0}" != 1 ]]; then
        echo "--- deleting GPU resources"
        delete_owned
    fi
}
trap cleanup EXIT

check_ray

echo "--- applying manifests"
kubectl -n "$NS" get secret hf-hopt0-token >/dev/null
apply_manifests "$ROOT/k8s/service/ray-head.yaml" "$ROOT/k8s/service/models/hoptimus0/ray-worker.yaml" \
    "$ROOT/k8s/service/models/hoptimus0/ray-worker-gpu.yaml" "$ROOT/k8s/service/api.yaml"
# A kept API pod still runs the old server.py; restart it to load the GPU route.
kubectl -n "$NS" rollout restart deploy/vhmodels-inference-api
kubectl -n "$NS" rollout status deploy/vhmodels-ray-head --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-ray-worker --timeout=10m
# The CUDA image is several GB, so the first pull on a node takes a while.
if ! kubectl -n "$NS" rollout status deploy/vhmodels-ray-worker-gpu --timeout=20m; then
    echo "--- GPU worker did not start; its pod and events:" >&2
    kubectl -n "$NS" describe pod -l app.kubernetes.io/name=vhmodels-ray-worker-gpu \
        | sed -n '/^Node:/p;/^Status:/p;/^Events:/,$p' >&2
    exit 1
fi
kubectl -n "$NS" rollout status deploy/vhmodels-inference-api --timeout=5m

echo "--- GPU worker: nvidia-smi"
kubectl -n "$NS" get pod -l app.kubernetes.io/name=vhmodels-ray-worker-gpu \
    -o jsonpath='{.items[0].spec.nodeName}{"\n"}'
kubectl -n "$NS" exec deploy/vhmodels-ray-worker-gpu -- nvidia-smi

echo "--- port-forwarding"
port_forward svc/vhmodels-ray-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT
port_forward svc/vhmodels-inference-api 8000
API=http://localhost:$LOCAL_PORT

echo "--- Ray jobs: start the CPU and the GPU actor (the GPU one installs ~3GB of CUDA torch)"
"$RAY" job submit --address "$DASHBOARD" --working-dir "$ROOT/k8s/service/models/hoptimus0" -- python deploy.py &
CPU=$!
"$RAY" job submit --address "$DASHBOARD" --working-dir "$ROOT/k8s/service/models/hoptimus0" -- python deploy.py --gpu &
GPU=$!
wait "$CPU"
wait "$GPU"

echo "--- CPU vs GPU"
python3 "$HERE/compare.py" "$API"

echo "--- load test against the GPU actor"
python3 "$ROOT/k8s/tests/two-models/load.py" "$API" "${DURATION:-60}" "${CONCURRENCY:-4}" hoptimus0-gpu

echo "--- GPU example passed"
