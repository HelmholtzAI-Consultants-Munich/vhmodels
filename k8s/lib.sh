# Shared helpers for the k8s example scripts. Source it after setting NS and
# ROOT, then call check_ray before using $RAY.

RAY_VERSION=2.59.0
RAY=${RAY:-$ROOT/.venv/bin/ray}
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

# Applies the manifests and remembers the ones this run created. Manifests whose
# objects already existed belong to a running deployment (all tests share the
# service's Ray head and API), so delete_owned keeps them.
apply_manifests() {
    local args=() f
    for f in "$@"; do
        kubectl -n "$NS" get -f "$f" >/dev/null 2>&1 || OWNED+=("$f")
        args+=(-f "$f")
    done
    kubectl -n "$NS" apply "${args[@]}"
}

delete_owned() {
    local args=() f
    for f in ${OWNED[@]+"${OWNED[@]}"}; do
        args+=(-f "$f")
    done
    if [[ ${#args[@]} -gt 0 ]]; then
        kubectl -n "$NS" delete "${args[@]}" --ignore-not-found --wait=false
    else
        echo "nothing to delete: everything was already running before this run"
    fi
}

stop_port_forwards() {
    local pid
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}

# wait_for_node_label DASHBOARD KEY=VALUE: waits until an alive Ray node has the
# label, e.g. a model's image worker (k8s/service/workers), so that a job does
# not wait forever on an actor whose node never joined.
wait_for_node_label() {
    local i
    for i in $(seq 60); do
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

# port_forward SERVICE PORT: forwards the service port to a free local port and
# sets LOCAL_PORT, so a tunnel the user already has open is never reused.
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
