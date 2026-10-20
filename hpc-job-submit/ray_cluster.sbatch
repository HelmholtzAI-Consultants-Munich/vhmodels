#!/bin/bash
#SBATCH --job-name=ray_cluster
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_3g.20gb:4
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
# symmetric-run starts `ray`, and job entrypoints run `python`, from this environment.
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
DRIVER_ENV="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
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

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
echo "Ray cluster on ${SLURM_JOB_NODELIST} with ${SLURM_GPUS_ON_NODE} GPU(s) per node; submit jobs with:"
echo "  RAY_AUTH_MODE=token RAY_ADDRESS=http://${HEAD_NODE}:8265 ray job submit -- python <script>"

# The entrypoint only keeps the cluster up until the job is cancelled or times out.
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" \
    ray symmetric-run \
    --address "${HEAD_NODE}:6379" \
    --min-nodes "${SLURM_JOB_NUM_NODES}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_ON_NODE}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats \
    -- sleep infinity
