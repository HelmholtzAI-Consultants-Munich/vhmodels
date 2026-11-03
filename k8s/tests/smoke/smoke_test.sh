#!/usr/bin/env bash
# Port-forward smoke test for the chc namespace:
#   1. deploy a Ray head and a placeholder API,
#   2. port-forward Ray Jobs (8265) and the API (8000) to free local ports,
#   3. curl the API and submit job.py with `ray job submit`,
#   4. delete what this run created (KEEP=1 leaves it running). A Ray head that
#      was already running, e.g. from KEEP=1 k8s/tests/hoptimus0/run.sh, is reused
#      and kept.
#
# Needs: kubectl with KUBECONFIG set, and the repo's .venv with
# ray[default]==2.59.0 (k8s/README.md, Setup); override with RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/k8s/lib.sh"

cleanup() {
    stop_port_forwards
    if [[ "${KEEP:-0}" != 1 ]]; then
        echo "--- deleting smoke resources"
        delete_owned
    fi
}
trap cleanup EXIT

check_ray

echo "--- applying manifests"
apply_manifests "$ROOT/k8s/service/ray-head.yaml" "$HERE/api.yaml"
kubectl -n "$NS" rollout status deploy/vhmodels-ray-head --timeout=10m
kubectl -n "$NS" rollout status deploy/vhmodels-api-smoke --timeout=5m

echo "--- port-forwarding"
port_forward svc/vhmodels-ray-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT
port_forward svc/vhmodels-api-smoke 8000
API=http://localhost:$LOCAL_PORT

echo "--- API: GET $API/health"
curl -fsS "$API/health"
echo

echo "--- Ray Jobs: ray job submit --address $DASHBOARD"
"$RAY" job submit --address "$DASHBOARD" --working-dir "$HERE" -- python job.py

echo "--- smoke test passed"
