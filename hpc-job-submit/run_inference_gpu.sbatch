#!/bin/bash
#SBATCH --job-name=inference_gpu
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
    echo "Usage: sbatch run_inference_gpu.sbatch /absolute/path/config.json" >&2
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

# The Python environment containing vhmodels runs the host-side proxy.
PYTHON_BIN="${VHMODELS_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
APPTAINER_BIN="/usr/bin/apptainer"

# Inference configuration shared with build_apptainer.sbatch.
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Keep Hub model files in the persistent Lustre cache. 
unset HF_HOME
export HF_HUB_CACHE="$(jq -er '.HF_CACHE_DIR' "${CONFIG_FILE}")/hub"
export DISABLE_SAFETENSORS_CONVERSION=true
mkdir -p "${HF_HUB_CACHE}"
HF_HUB_CACHE="$(realpath -e "${HF_HUB_CACHE}")"

# Persistent image produced by build_apptainer.sbatch.
IMAGE_PATH="${IMAGE_DIR}/vhmodels-${MODEL}.sif"
export VHMODELS_IMAGE_PATH="${IMAGE_PATH}"

# Request NVIDIA driver/device passthrough and explicit CUDA model placement.
export APPTAINER_NV=1
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
OUTPUT_DIR="$(realpath -e "${OUTPUT_DIR}")"
export VHMODELS_OUTPUT_DIR="${OUTPUT_DIR}"

unset VHMODELS_DATA_DIR
DATA_DIR="$(jq -r '.DATA_DIR // empty' "${CONFIG_FILE}")"
if [[ -n "${DATA_DIR}" ]]; then
    if [[ "${DATA_DIR}" != /* || ! -d "${DATA_DIR}" ]]; then
        echo "ERROR: DATA_DIR must be an absolute path to an existing directory." >&2
        exit 1
    fi
    export VHMODELS_DATA_DIR="$(realpath -e "${DATA_DIR}")"
fi

INFERENCE_DIR="$(dirname "${INFERENCE_SCRIPT}")"
BINDS="${OUTPUT_DIR}:${OUTPUT_DIR},${INFERENCE_DIR}:${INFERENCE_DIR},${HF_HUB_CACHE}:${HF_HUB_CACHE}"
if [[ -n "${VHMODELS_DATA_DIR:-}" ]]; then
    BINDS+=",${VHMODELS_DATA_DIR}:${VHMODELS_DATA_DIR}"
fi
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}${BINDS}"
cd "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c 'import vhmodels; print("Using vhmodels from", vhmodels.__file__)'; then
    echo "ERROR: vhmodels is not importable by ${PYTHON_BIN}." >&2
    exit 1
fi

[[ -x "${APPTAINER_BIN}" ]] || {
    echo "ERROR: ${APPTAINER_BIN} is unavailable." >&2
    exit 1
}
[[ -f "${IMAGE_PATH}" ]] || {
    echo "ERROR: ${IMAGE_PATH} does not exist. Run build_apptainer.sbatch first." >&2
    exit 1
}

"${APPTAINER_BIN}" --version

# Fail early if the allocated GPU is not visible inside the container.
srun --ntasks=1 --gpus=1 \
    "${APPTAINER_BIN}" exec --nv "${IMAGE_PATH}" \
    /opt/venv/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable inside Apptainer"; print(f"GPU: {torch.cuda.get_device_name(0)}")'

echo "Running ${MODEL} inference script ${INFERENCE_SCRIPT} on ${VHMODELS_DEVICE} with ${IMAGE_PATH}"
srun --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus=1 \
    "${PYTHON_BIN}" "${INFERENCE_SCRIPT}"
