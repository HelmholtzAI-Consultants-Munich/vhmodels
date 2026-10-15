#!/bin/bash
#SBATCH --job-name=inference_ray
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --acctg-freq=task=1
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch run_inference_ray.sbatch /absolute/path/config.json" >&2
    exit 1
fi
CONFIG_FILE="$1"
if [[ "${CONFIG_FILE}" != /* || ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Pass an absolute path to an existing config.json." >&2
    exit 1
fi
CONFIG_FILE="$(realpath -e "${CONFIG_FILE}")"
INFERENCE_SCRIPT="$(jq -er '.INFERENCE_SCRIPT' "${CONFIG_FILE}")"
if [[ "${INFERENCE_SCRIPT}" != /* || ! -f "${INFERENCE_SCRIPT}" ]]; then
    echo "ERROR: INFERENCE_SCRIPT must be an absolute path to a Python file." >&2
    exit 1
fi
INFERENCE_SCRIPT="$(realpath -e "${INFERENCE_SCRIPT}")"

# Remove inherited Conda activation markers from the host controller.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

# The Ray driver Python runs the script and the Ray head.
PYTHON_BIN="${VHMODELS_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
RAY_BIN="$(dirname "${PYTHON_BIN}")/ray"
[[ -x "${RAY_BIN}" ]] || {
    echo "ERROR: ${RAY_BIN} is unavailable; install vhmodels with the 'ray' extra." >&2
    exit 1
}

MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Model weights go to the shared caches from the config. HF_HOME stays at its
# default (~/.cache/huggingface) so the Hugging Face login token is found.
unset HF_HOME
export HF_HUB_CACHE="$(jq -er '.HF_CACHE_DIR' "${CONFIG_FILE}")/hub"
export TORCH_HOME="$(jq -er '.TORCH_CACHE_DIR' "${CONFIG_FILE}")"
export DISABLE_SAFETENSORS_CONVERSION=true
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}"

export VHMODELS_RUNTIME="ray"
export VHMODELS_VENV_DIR="$(jq -er '.VENV_DIR' "${CONFIG_FILE}")"
# A cuda device makes the model's actor request one GPU.
export VHMODELS_DEVICE="cuda:0"

# Use a stable working directory regardless of where sbatch was called.
OUTPUT_DIR="$(jq -r '.OUTPUT_DIR // empty' "${CONFIG_FILE}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="$(dirname "${CONFIG_FILE}")/output"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    echo "ERROR: OUTPUT_DIR must be an absolute path." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"
export VHMODELS_OUTPUT_DIR="$(realpath -e "${OUTPUT_DIR}")"

unset VHMODELS_DATA_DIR
DATA_DIR="$(jq -r '.DATA_DIR // empty' "${CONFIG_FILE}")"
if [[ -n "${DATA_DIR}" ]]; then
    if [[ "${DATA_DIR}" != /* || ! -d "${DATA_DIR}" ]]; then
        echo "ERROR: DATA_DIR must be an absolute path to an existing directory." >&2
        exit 1
    fi
    export VHMODELS_DATA_DIR="$(realpath -e "${DATA_DIR}")"
fi
cd "${VHMODELS_OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c 'import vhmodels; print("Using vhmodels from", vhmodels.__file__)'; then
    echo "ERROR: vhmodels is not importable by ${PYTHON_BIN}." >&2
    exit 1
fi

# Single-node Ray head; ports derive from the job id to avoid collisions on shared nodes.
NUM_GPUS="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")"
HEAD_IP="$(hostname -i | awk '{print $1}')"
RAY_PORT=$((20000 + (SLURM_JOB_ID % 1000) * 10))
# Ray's socket paths must stay under 108 bytes (see docs/ray.md).
RAY_TMP_DIR="/localscratch/${USER}/ray-${SLURM_JOB_ID}"
if (( ${#RAY_TMP_DIR} > 40 )); then
    RAY_TMP_DIR="/tmp/ray-${SLURM_JOB_ID}"
fi
mkdir -p "${RAY_TMP_DIR}"
# Connecting drivers only send the cluster token in token mode (see docs/ray.md).
export RAY_AUTH_MODE=token
"${RAY_BIN}" start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --ray-client-server-port=$((RAY_PORT + 1)) \
    --dashboard-agent-listen-port=$((RAY_PORT + 2)) \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${NUM_GPUS}" \
    --temp-dir="${RAY_TMP_DIR}" \
    --include-dashboard=false \
    --disable-usage-stats
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Running ${MODEL} inference script ${INFERENCE_SCRIPT} on Ray at ${RAY_ADDRESS} with ${NUM_GPUS} GPU(s)"
"${PYTHON_BIN}" "${INFERENCE_SCRIPT}"
