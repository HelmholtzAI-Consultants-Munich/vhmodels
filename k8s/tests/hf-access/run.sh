#!/usr/bin/env bash
# Runs the H-optimus-0 access check as a Job in the chc namespace, prints its
# output and deletes it again (KEEP=1 leaves it). Needs the hf-hopt0-token
# Secret with key HF_TOKEN.
set -euo pipefail

NS=${NS:-chc}
HERE=$(cd "$(dirname "$0")" && pwd)

cleanup() {
    if [[ "${KEEP:-0}" != 1 ]]; then
        kubectl -n "$NS" delete -f "$HERE/hf-access-job.yaml" --ignore-not-found --wait=false >/dev/null
    fi
}
trap cleanup EXIT

kubectl -n "$NS" get secret hf-hopt0-token >/dev/null
kubectl -n "$NS" delete job vhmodels-hf-access --ignore-not-found --wait=true >/dev/null
kubectl -n "$NS" apply -f "$HERE/hf-access-job.yaml"

for _ in $(seq 150); do
    succeeded=$(kubectl -n "$NS" get job vhmodels-hf-access -o jsonpath='{.status.succeeded}')
    failed=$(kubectl -n "$NS" get job vhmodels-hf-access -o jsonpath='{.status.failed}')
    [[ -n "$succeeded" || -n "$failed" ]] && break
    sleep 2
done

echo "--- job output"
kubectl -n "$NS" logs job/vhmodels-hf-access

if [[ "${succeeded:-}" == 1 ]]; then
    echo "--- HF access test passed"
else
    echo "--- HF access test FAILED" >&2
    kubectl -n "$NS" describe job vhmodels-hf-access | tail -n 15 >&2
    exit 1
fi
