#!/usr/bin/env bash
# The models from their GHCR images, with vhmodels as the images ship it:
#   1. deploy the Ray head (vhmodels-head image), a worker node per model
#      (models/<model id>.yaml) and the Serve config,
#   2. per model, Ray Serve: wait for its app and send it the example input,
#   3. per model, Ray Core: its example script as a Ray job, or the same call
#      in job.py where the script does not fit the pod (test_spec),
#   4. delete what this run created (KEEP=1 leaves it running). Resources that
#      were already running are kept, and so are the model pods' volumes with
#      the downloaded sources (cache-vhmodels-<model id>-0).
# Serve comes first: its replica downloads the sources once, and the job's
# model then loads next to a loaded replica instead of at the same time, which
# H-Optimus-0's memory would not fit. A failed model does not stop the others.
#
# Usage: k8s/run.sh [MODEL ...]; without models, every file in models/.
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
APPLIED=
SERVE=

# Prints the size of a Serve response's output: predictions or embeddings.
SUMMARY='import json, sys
output = json.load(open(sys.argv[1]))["output"]
if isinstance(output, dict):
    print(len(output), "predictions, the first five:")
    for key, value in list(output.items())[:5]:
        print(f"  {key}\t{value}")
else:
    shape = []
    while isinstance(output, list):
        shape.append(len(output))
        output = output[0] if output else None
    print("output of shape", "x".join(map(str, shape)))'

# test_spec MODEL: sets the model's test,
#   FILES    the job's files, copied to its working_dir at their repository paths
#   SCRIPT   the source of job.py, for models without a usable examples/ script
#   JOB      the job's command
#   ROUTE    the Serve route, and REQUEST the curl arguments of its request
test_spec() {
    FILES=()
    SCRIPT=
    case $1 in
        mole)
            FILES=(examples/mole.py example_data/MolE/examples_molecules.tsv)
            JOB='python examples/mole.py && head -n 6 output/mole_predictions.tsv'
            ROUTE=/mole/predict/files
            REQUEST=(-F "input=@$ROOT/${FILES[1]}")
            ;;
        dinobloom)
            FILES=(examples/dinobloom.py example_data/DinoBloom/001.bmp)
            JOB='python examples/dinobloom.py'
            ROUTE=/dinobloom/embed/files
            REQUEST=(-F "input=@$ROOT/${FILES[1]}")
            ;;
        hoptimus0)
            # examples/hoptimus0.py on the half of the GPU that Serve leaves
            # (serve.yaml).
            FILES=(example_data/H-Optimus-0/TUM-AACHEHDV.tif)
            read -r -d '' SCRIPT <<EOF || true
import vhmodels

with vhmodels.load_model(project="hoptimus0", runtime="ray", device="cuda:0",
                         ray_options={"num_gpus": 0.5}) as model:
    print("GPU", model.info()["cuda_visible_devices"])
    embeddings = model.embed(input="${FILES[0]}", batch_size=1)
print(len(embeddings), "embedding of dimension", len(embeddings[0]))
EOF
            JOB='python job.py'
            ROUTE=/hoptimus0/embed/files
            REQUEST=(-F "input=@$ROOT/${FILES[0]}")
            ;;
        hyformer)
            FILES=(examples/hyformer.py example_data/Hyformer/molecules.smi
                example_data/Hyformer/peptides.txt)
            JOB='python examples/hyformer.py'
            ROUTE=/hyformer/embed
            REQUEST=(-H 'Content-Type: application/json' -d '{"input": ["CCO", "c1ccccc1O"]}')
            ;;
        nicheformer)
            # No script in examples/: the call of the repository's test.py.
            FILES=(example_data/Nicheformer/xenium_mean_script.npy
                example_data/Nicheformer/preprocessed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_sample-1000.h5ad)
            read -r -d '' SCRIPT <<EOF || true
import vhmodels

with vhmodels.load_model(project="nicheformer", runtime="ray") as model:
    embeddings = model.embed(
        input={"technology_mean": "${FILES[0]}", "data": "${FILES[1]}"},
        batch_size=4,
        max_cells=4,
    )
print(len(embeddings), "embeddings of dimension", len(embeddings[0]))
EOF
            JOB='python job.py'
            ROUTE=/nicheformer/embed/files
            REQUEST=(-F "technology_mean=@$ROOT/${FILES[0]}" -F "data=@$ROOT/${FILES[1]}"
                -F 'kwargs={"max_cells": 4}')
            ;;
        prottrans)
            # examples/prottrans.py uses prot_t5_xxl_bfd, which does not fit the
            # pod (models/prottrans.yaml): the same call with Serve's variant.
            read -r -d '' SCRIPT <<'EOF' || true
import vhmodels

with vhmodels.load_model(project="prottrans", model="prot_bert", runtime="ray") as model:
    embeddings = model.embed(input=["PRTEINO", "SEQWENCE"])
print(len(embeddings), "embeddings of", [len(embedding) for embedding in embeddings], "residues")
EOF
            JOB='python job.py'
            ROUTE=/prottrans/embed
            REQUEST=(-H 'Content-Type: application/json' -d '{"input": ["PRTEINO", "SEQWENCE"]}')
            ;;
        *)
            echo "run.sh has no test for '$1'; add one to test_spec" >&2
            return 1
            ;;
    esac
}

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

# check_secrets MANIFEST: the Secrets a worker reads must exist, or its pod
# never starts.
check_secrets() {
    local secret
    for secret in $(sed -n 's/.*secretKeyRef: *{name: *\([^,} ]*\).*/\1/p' "$1"); do
        if ! kubectl -n "$NS" get secret "$secret" >/dev/null 2>&1; then
            echo "$(basename "$1") reads the Secret $secret, which does not exist; see k8s/README.md (Setup)" >&2
            return 1
        fi
    done
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

# wait_for_app DASHBOARD APP: waits until the Serve app is RUNNING, or failed.
wait_for_app() {
    local status=NONE
    for _ in $(seq 120); do
        status=$(curl -fsS "$1/api/serve/applications/" 2>/dev/null | python3 -c 'import json, sys
app = json.load(sys.stdin)["applications"].get(sys.argv[1], {})
print(app.get("status", "NONE"), app.get("message", ""))' "$2") || status=UNREACHABLE
        # UNHEALTHY is not final: an app whose worker pod was just replaced
        # is UNHEALTHY until Serve restarts its replica on the new pod.
        case ${status%% *} in
            RUNNING) return 0 ;;
            DEPLOY_FAILED) break ;;
        esac
        sleep 5
    done
    echo "app $2: $status" >&2
    return 1
}

# test_model MODEL: Ray Serve, then Ray Core. It runs with errexit off (it is
# called with ||), so every step returns on failure itself.
test_model() {
    local model=$1 dir=$STAGE/$1 file
    test_spec "$model" || return 1
    mkdir -p "$dir" || return 1
    for file in ${FILES[@]+"${FILES[@]}"}; do
        mkdir -p "$dir/$(dirname "$file")" && cp "$ROOT/$file" "$dir/$file" || return 1
    done
    [[ -z $SCRIPT ]] || printf '%s\n' "$SCRIPT" >"$dir/job.py" || return 1

    echo "--- $model: waiting for its pod and its Ray node"
    kubectl -n "$NS" rollout status "statefulset/vhmodels-$model" --timeout=15m || return 1
    wait_for_node_label "$DASHBOARD" "vhmodels/project=$model" || return 1

    echo "--- $model: Ray Serve, waiting for the app (its first start downloads the sources)"
    wait_for_app "$DASHBOARD" "$model" || return 1
    if [[ -z $SERVE ]]; then
        # Only now: kubectl port-forward exits if the port is closed on first use.
        port_forward svc/vhmodels-head 8000 || return 1
        SERVE=http://localhost:$LOCAL_PORT
    fi
    echo "--- $model: Ray Serve, POST $ROUTE"
    if ! curl -sS --fail-with-body --max-time 600 -o "$STAGE/$model.json" \
        "${REQUEST[@]}" "$SERVE$ROUTE"; then
        cat "$STAGE/$model.json" >&2 2>/dev/null
        echo >&2
        return 1
    fi
    python3 -c "$SUMMARY" "$STAGE/$model.json" || return 1

    echo "--- $model: Ray Core, $JOB"
    "$RAY" job submit --address "$DASHBOARD" --working-dir "$dir" -- bash -c "$JOB" || return 1
    echo "--- $model: passed"
}

cleanup() {
    local pid args=() f
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$STAGE"
    [[ -z $APPLIED || "${KEEP:-0}" == 1 ]] && return
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

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
    for f in "$HERE"/models/*.yaml; do
        MODELS+=("$(basename "$f" .yaml)")
    done
fi
MANIFESTS=("$HERE/serve.yaml" "$HERE/head.yaml")
for model in "${MODELS[@]}"; do
    manifest=$HERE/models/$model.yaml
    if [[ ! -f $manifest ]]; then
        echo "no worker for '$model': k8s/models/$model.yaml does not exist" >&2
        exit 1
    fi
    test_spec "$model"
    check_secrets "$manifest"
    MANIFESTS+=("$manifest")
done

echo "--- applying manifests for ${MODELS[*]} (a node without a model's image pulls it first, ~5 min)"
APPLIED=1
apply_manifests "${MANIFESTS[@]}"
kubectl -n "$NS" rollout status deploy/vhmodels-head --timeout=15m

# A head that was already running has the Serve config it started with.
echo "--- deploying serve.yaml"
kubectl -n "$NS" get configmap vhmodels-serve -o jsonpath='{.data.serve\.yaml}' \
    | kubectl -n "$NS" exec -i deploy/vhmodels-head -- \
        bash -c 'cat >/tmp/serve.yaml && serve deploy /tmp/serve.yaml'

echo "--- port-forwarding"
port_forward svc/vhmodels-head 8265
DASHBOARD=http://localhost:$LOCAL_PORT

FAILED=()
for model in "${MODELS[@]}"; do
    test_model "$model" || {
        echo "--- $model: FAILED" >&2
        FAILED+=("$model")
    }
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "--- failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "--- passed: ${MODELS[*]}"
