#!/bin/bash
#SBATCH --job-name=inference_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --acctg-freq=task=1
#SBATCH --time=1:00:00
#SBATCH --output=./output/inference_gpu_job_output.txt
#SBATCH --error=./output/inference_gpu_job_error.txt
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -euo pipefail

# Require submission from the repository root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ ! -f "${PROJECT_ROOT}/test.py" ]]; then
    echo "ERROR: Submit this job from the vhmodels repository root." >&2
    exit 1
fi

# Remove inherited Conda activation markers from the host controller.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

# System Python runs the proxy; Apptainer runs the GPU model worker.
PYTHON_BIN="/usr/bin/python3"
APPTAINER_BIN="/usr/bin/apptainer"

# Inference configuration shared with build_apptainer.sbatch.
CONFIG_FILE="${PROJECT_ROOT}/hpc-job-submit/config.json"
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Keep Hub model files in the persistent Lustre cache. 
unset HF_HOME
export HF_HUB_CACHE="$(jq -er '.HF_CACHE_DIR' "${CONFIG_FILE}")/hub"
export DISABLE_SAFETENSORS_CONVERSION=true
mkdir -p "${HF_HUB_CACHE}"

# Persistent image produced by build_apptainer.sbatch.
IMAGE_PATH="${IMAGE_DIR}/vhmodels-${MODEL}.sif"
export VHMODELS_IMAGE_PATH="${IMAGE_PATH}"

# Request NVIDIA driver/device passthrough and explicit CUDA model placement.
export APPTAINER_NV=1
export VHMODELS_DEVICE="cuda:0"

# Expose the repository, example inputs, and persistent model cache.
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}${PROJECT_ROOT}:${PROJECT_ROOT},${HF_HUB_CACHE}:${HF_HUB_CACHE}"

cd "${PROJECT_ROOT}"

[[ -x "${PYTHON_BIN}" ]] || {
    echo "ERROR: ${PYTHON_BIN} is unavailable." >&2
    exit 1
}
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

echo "Running ${MODEL} test.py on ${VHMODELS_DEVICE} with ${IMAGE_PATH}"
srun --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus=1 \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/test.py"
