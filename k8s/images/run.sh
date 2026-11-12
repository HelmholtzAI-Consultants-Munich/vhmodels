#!/usr/bin/env bash
# MolE from its GHCR ray image, with vhmodels as the image ships it:
#   1. deploy the Ray head (model-free head image) and a MolE worker node,
#   2. Ray Core: the unchanged examples/mole.py as a Ray job,
#   3. Ray Serve: wait for /mole, which the head deploys from serve.yaml on
#      start, and POST the same table to it,
#   4. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running are kept, and so is the MolE pod's volume with the
#      downloaded sources (cache-vhmodels-mole-0).
#
# Needs the repo's .venv with Ray 2.59.0 (k8s/README.md, Setup); override with
# RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
source "$ROOT/k8s/lib.sh"
STAGE=$(mktemp -d)

cleanup() {
    stop_port_forwards
    rm -rf "$STAGE"
    if [[ "${KEEP:-0}" != 1 ]]; then
        echo "--- deleting inference resources"
        delete_owned
    fi
}
trap cleanup EXIT

check_ray

echo "--- applying manifests (a node without the MolE image pulls it first, ~5 min)"
apply_manifests "$HERE/serve.yaml" "$HERE/head.yaml" "$HERE/mole.yaml"
kubectl -n "$NS" rollout status deploy/vhmodels-head --timeout=15m
kubectl -n "$NS" rollout status statefulset/vhmodels-mole --timeout=15m

echo "--- port-forwarding"
port_forward svc/vhmodels-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT

echo "--- waiting for the MolE node to join the cluster"
wait_for_node_label "$DASHBOARD" vhmodels/project=mole

# The job's working_dir: the example and its input, at their repository paths.
mkdir -p "$STAGE/examples" "$STAGE/example_data/MolE"
cp "$ROOT/examples/mole.py" "$STAGE/examples/"
cp "$ROOT/example_data/MolE/examples_molecules.tsv" "$STAGE/example_data/MolE/"

echo "--- Ray Core: examples/mole.py (the first run downloads the MolE sources)"
"$RAY" job submit --address "$DASHBOARD" --working-dir "$STAGE" \
    -- bash -c 'python examples/mole.py && head -n 6 output/mole_predictions.tsv'

echo "--- Ray Serve: waiting for the app the head deployed from serve.yaml"
for _ in $(seq 60); do
    STATUS=$(curl -fsS "$DASHBOARD/api/serve/applications/" | python3 -c 'import json, sys
print(json.load(sys.stdin)["applications"].get("mole", {}).get("status", "NONE"))')
    [[ $STATUS == RUNNING || $STATUS == DEPLOY_FAILED ]] && break
    sleep 5
done
echo "app mole: $STATUS"
[[ $STATUS == RUNNING ]]

# Only now: kubectl port-forward exits if the port is closed on first use.
port_forward svc/vhmodels-head 8000
SERVE=http://localhost:$LOCAL_PORT

# MolE takes a file path, which must exist on the replica's pod.
echo "--- Ray Serve: POST the table to $SERVE/mole/predict"
kubectl -n "$NS" cp "$ROOT/example_data/MolE/examples_molecules.tsv" \
    vhmodels-mole-0:/cache/examples_molecules.tsv
curl -fsS -H 'Content-Type: application/json' \
    -d '{"input": "/cache/examples_molecules.tsv"}' "$SERVE/mole/predict" \
    | python3 -c 'import json, sys
predictions = json.load(sys.stdin)["output"]
print(len(predictions), "predictions, the first five:")
for pred_id, probability in list(predictions.items())[:5]:
    print(f"  {pred_id}\t{probability:.4f}")'

echo "--- MolE test passed"
