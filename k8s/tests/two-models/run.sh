#!/usr/bin/env bash
# Two models served side by side on the chc namespace, CPU only:
#   1. deploy the Ray head, the H-Optimus-0 worker, a second worker for
#      ProtT5-XL-UniRef50 and the inference API,
#   2. submit both deploy jobs, which start the two model actors,
#   3. run load.py: each model alone, then both under concurrent load,
#   4. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running, e.g. a kept H-Optimus-0 test, are reused and kept.
#
# DURATION (default 60s) and CONCURRENCY (default 4 clients per model) set the
# concurrent phase. Needs the hf-hopt0-token Secret and the repo's .venv with
# ray[default]==2.59.0 (k8s/README.md, Setup); override with RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/k8s/lib.sh"

cleanup() {
    stop_port_forwards
    if [[ "${KEEP:-0}" != 1 ]]; then
        echo "--- deleting two-model resources"
        delete_owned
    fi
}
trap cleanup EXIT

check_ray

echo "--- applying manifests"
kubectl -n "$NS" get secret hf-hopt0-token >/dev/null
apply_manifests "$ROOT/k8s/service/ray-head.yaml" "$ROOT/k8s/service/models/hoptimus0/ray-worker.yaml" \
    "$ROOT/k8s/service/models/prottrans/ray-worker.yaml" "$ROOT/k8s/service/api.yaml"
# A kept API pod still runs the old server.py; restart it to load the
# two-model routes from the ConfigMap.
kubectl -n "$NS" rollout restart deploy/vhmodels-inference-api
kubectl -n "$NS" rollout status deploy/vhmodels-ray-head --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-ray-worker --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-ray-worker-prottrans --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-inference-api --timeout=5m

echo "--- port-forwarding"
port_forward svc/vhmodels-ray-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT
port_forward svc/vhmodels-inference-api 8000
API=http://localhost:$LOCAL_PORT

echo "--- Ray jobs: start both model actors (first run downloads ~4.5GB and ~11.3GB)"
"$RAY" job submit --address "$DASHBOARD" --working-dir "$ROOT/k8s/service/models/hoptimus0" -- python deploy.py &
HOPT=$!
"$RAY" job submit --address "$DASHBOARD" --working-dir "$ROOT/k8s/service/models/prottrans" -- python deploy.py &
PROT=$!
wait "$HOPT"
wait "$PROT"

echo "--- load test against $API"
python3 "$HERE/load.py" "$API" "${DURATION:-60}" "${CONCURRENCY:-4}"

echo "--- two-model example passed"
