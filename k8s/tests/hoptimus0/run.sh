#!/usr/bin/env bash
# End-to-end H-Optimus-0 inference on the chc namespace, CPU only:
#   1. deploy the Ray head, one model worker and the inference API (k8s/service),
#   2. submit deploy.py as a Ray job, which starts the model actor,
#   3. POST an example tile to the API through a port-forward,
#   4. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running, e.g. a Ray head from the smoke test, are kept.
#
# Needs the hf-hopt0-token Secret and the repo's .venv with
# ray[default]==2.59.0 (k8s/README.md, Setup); override with RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/k8s/lib.sh"
TILE=${TILE:-$ROOT/example_data/H-Optimus-0/TUM-AACHEHDV.tif}

cleanup() {
    stop_port_forwards
    if [[ "${KEEP:-0}" != 1 ]]; then
        echo "--- deleting inference resources"
        delete_owned
    fi
}
trap cleanup EXIT

check_ray

echo "--- applying manifests"
kubectl -n "$NS" get secret hf-hopt0-token >/dev/null
apply_manifests "$ROOT/k8s/service/ray-head.yaml" "$ROOT/k8s/service/models/hoptimus0/ray-worker.yaml" "$ROOT/k8s/service/api.yaml"
kubectl -n "$NS" rollout status deploy/vhmodels-ray-head --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-ray-worker --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-inference-api --timeout=5m

echo "--- port-forwarding"
port_forward svc/vhmodels-ray-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT
port_forward svc/vhmodels-inference-api 8000
API=http://localhost:$LOCAL_PORT

echo "--- Ray job: start the model actor (first run installs torch and downloads ~4.5GB)"
"$RAY" job submit --address "$DASHBOARD" --working-dir "$ROOT/k8s/service/models/hoptimus0" -- python deploy.py

echo "--- API: POST $(basename "$TILE") to $API/embed"
curl -fsS --data-binary @"$TILE" "$API/embed" \
    | python3 -c 'import json, sys; r = json.load(sys.stdin); print("dim", r["dim"], "first values", [round(v, 4) for v in r["embedding"][:5]])'

echo "--- H-Optimus-0 test passed"
