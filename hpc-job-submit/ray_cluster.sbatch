#!/bin/bash
#SBATCH --job-name=ray_cluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_20gb|a100_40gb|a100_80gb|v100_32gb|h100_80gb
#SBATCH --acctg-freq=task=1
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail

CONFIG_FILE="${1:?Usage: sbatch ray_cluster.sbatch /absolute/path/config.json}"
if [[ "${CONFIG_FILE}" != /* || ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Pass an absolute path to an existing config.json." >&2
    exit 1
fi

# Remove inherited Conda activation markers from the host controller.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

PYTHON_BIN="${VHMODELS_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
# Ray and job entrypoints (`python`) run from this environment.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
command -v ray >/dev/null 2>&1 || {
    echo "ERROR: ray is unavailable; install vhmodels with the 'ray' extra." >&2
    exit 1
}

# Jobs and model actors inherit everything below from the cluster.
unset HF_HOME
export HF_HUB_CACHE="$(jq -er '.HF_CACHE_DIR' "${CONFIG_FILE}")/hub"
export TORCH_HOME="$(jq -er '.TORCH_CACHE_DIR' "${CONFIG_FILE}")"
export DISABLE_SAFETENSORS_CONVERSION=true
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}"

export VHMODELS_RUNTIME="ray"
export VHMODELS_DEVICE="cuda:0"
export VHMODELS_VENV_DIR="$(jq -r '.VENV_DIR // empty' "${CONFIG_FILE}")"
export VHMODELS_IMAGE_DIR="$(jq -r '.IMAGE_DIR // empty' "${CONFIG_FILE}")"

OUTPUT_DIR="$(jq -r '.OUTPUT_DIR // empty' "${CONFIG_FILE}")"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CONFIG_FILE}")/output}"
mkdir -p "${OUTPUT_DIR}"
export VHMODELS_OUTPUT_DIR="$(realpath -e "${OUTPUT_DIR}")"
# Workers in model images run Ray's entrypoint from the driver env.
DRIVER_ENV="$(realpath -e "$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')")"
BINDS="${DRIVER_ENV},${VHMODELS_OUTPUT_DIR},${HF_HUB_CACHE},${TORCH_HOME}"

unset VHMODELS_DATA_DIR
DATA_DIR="$(jq -r '.DATA_DIR // empty' "${CONFIG_FILE}")"
if [[ -n "${DATA_DIR}" ]]; then
    export VHMODELS_DATA_DIR="$(realpath -e "${DATA_DIR}")"
    BINDS+=",${VHMODELS_DATA_DIR}"
fi
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}${BINDS}"

# Connecting drivers only send the cluster token in token mode (see docs/ray.md).
export RAY_AUTH_MODE=token

# Shared nodes: ports and temp dir come from the job id (see Ray's Slurm guide).
# The login node reaches compute nodes only on ports 8000-9000 and 60001-64000.
RAY_PORT=$((20000 + (SLURM_JOB_ID % 1000) * 10))
WORKER_PORT=$((30000 + (SLURM_JOB_ID % 500) * 40))
DASHBOARD_PORT=$((8000 + SLURM_JOB_ID % 1000))
HEAD_NODE="$(hostname)"
echo "Ray cluster on ${HEAD_NODE} with ${SLURM_GPUS_ON_NODE} GPU(s); submit jobs with:"
echo "  RAY_AUTH_MODE=token RAY_ADDRESS=http://${HEAD_NODE}:${DASHBOARD_PORT} ray job submit -- python <script>"

ray start --head --block \
    --port="${RAY_PORT}" \
    --ray-client-server-port=$((RAY_PORT + 1)) \
    --dashboard-port="${DASHBOARD_PORT}" \
    --dashboard-agent-listen-port=$((RAY_PORT + 3)) \
    --min-worker-port="${WORKER_PORT}" \
    --max-worker-port=$((WORKER_PORT + 39)) \
    --temp-dir="/tmp/ray-${SLURM_JOB_ID}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_ON_NODE}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats
