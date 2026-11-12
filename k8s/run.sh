#!/usr/bin/env bash
# MolE from its GHCR images, with vhmodels as the images ship it:
#   1. deploy the Ray head (vhmodels-head image) and a MolE worker node,
#   2. Ray Core: the unchanged examples/mole.py as a Ray job,
#   3. Ray Serve: wait for /mole, which the head deploys from serve.yaml on
#      start, and upload the same table to /mole/predict/files,
#   4. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running are kept, and so is the MolE pod's volume with the
#      downloaded sources (cache-vhmodels-mole-0).
#
# Needs the repo's .venv with ray[default]==2.59.0 (k8s/README.md, Setup);
# override with RAY=/path/to/ray.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
RAY_VERSION=2.59.0
RAY=${RAY:-$ROOT/.venv/bin/ray}
STAGE=$(mktemp -d)
PIDS=()
OWNED=()

# The Ray CLI must be the cluster's version with the ray[default] extras, or
# `ray job` fails with "require the ray[default] installation".
check_ray() {
    if [[ ! -x "$RAY" ]]; then
        echo "no Ray CLI at $RAY: create .venv as in k8s/README.md (Setup), or set RAY=/path/to/ray" >&2
        return 1
    fi
    local python
    python=$(dirname "$RAY")/python
    [[ -x "$python" ]] || python=$(head -1 "$RAY" | sed 's/^#!//')
    if [[ ! -x "$python" ]]; then
        echo "the Python behind $RAY ($python) is missing: recreate .venv as in k8s/README.md (Setup)" >&2
        return 1
    fi
    if ! "$python" -c "import sys, aiohttp, requests, ray; sys.exit(ray.__version__ != '$RAY_VERSION')" 2>/dev/null; then
        echo "$RAY is not ray[default]==$RAY_VERSION; install it with:" >&2
        echo "    $python -m pip install 'ray[default]==$RAY_VERSION'" >&2
        return 1
    fi
}

# Applies the manifests and remembers the ones this run created. The others
# belong to a service that was already running, which cleanup keeps.
apply_manifests() {
    local args=() f
    for f in "$@"; do
        kubectl -n "$NS" get -f "$f" >/dev/null 2>&1 || OWNED+=("$f")
        args+=(-f "$f")
    done
    kubectl -n "$NS" apply "${args[@]}"
}

# port_forward SERVICE PORT: forwards the port to a free local port and sets
# LOCAL_PORT, so tunnels you already have open on 8265 or 8000 are not affected.
port_forward() {
    LOCAL_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    kubectl -n "$NS" port-forward "$1" "$LOCAL_PORT:$2" >/dev/null &
    PIDS+=($!)
    for _ in $(seq 30); do
        nc -z localhost "$LOCAL_PORT" 2>/dev/null && return 0
        sleep 1
    done
    echo "port-forward of $1 on $LOCAL_PORT did not come up" >&2
    return 1
}

# wait_for_node_label DASHBOARD KEY=VALUE: waits until an alive Ray node has the
# label, so that a job does not wait forever on an actor whose node never joined.
wait_for_node_label() {
    for _ in $(seq 60); do
        if "$RAY" list nodes --address "$1" --filter state=ALIVE --format json 2>/dev/null \
            | python3 -c 'import json, sys
key, value = sys.argv[1].split("=", 1)
sys.exit(not any(node["labels"].get(key) == value for node in json.load(sys.stdin)))' "$2"; then
            return 0
        fi
        sleep 5
    done
    echo "no alive Ray node with label $2 after 5 min; check the worker's logs" >&2
    return 1
}

cleanup() {
    local pid args=() f
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$STAGE"
    [[ "${KEEP:-0}" == 1 ]] && return
    echo "--- deleting inference resources"
    for f in ${OWNED[@]+"${OWNED[@]}"}; do
        args+=(-f "$f")
    done
    if [[ ${#args[@]} -gt 0 ]]; then
        kubectl -n "$NS" delete "${args[@]}" --ignore-not-found --wait=false
    else
        echo "nothing to delete: everything was already running before this run"
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

echo "--- Ray Serve: upload the table to $SERVE/mole/predict/files"
curl -fsS -F "input=@$ROOT/example_data/MolE/examples_molecules.tsv" \
    "$SERVE/mole/predict/files" \
    | python3 -c 'import json, sys
predictions = json.load(sys.stdin)["output"]
print(len(predictions), "predictions, the first five:")
for pred_id, probability in list(predictions.items())[:5]:
    print(f"  {pred_id}\t{probability:.4f}")'

echo "--- MolE test passed"
