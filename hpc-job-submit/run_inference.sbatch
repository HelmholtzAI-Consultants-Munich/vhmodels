#!/bin/bash
#SBATCH --job-name=inference_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --acctg-freq=task=1
#SBATCH --time=1:00:00
#SBATCH --output=./output/inference_cpu_job_output.txt
#SBATCH --error=./output/inference_cpu_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

# Require submission from the repository root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ ! -f "${PROJECT_ROOT}/test.py" ]]; then
    echo "ERROR: Submit this job from the vhmodels repository root." >&2
    exit 1
fi

# Remove inherited Conda activation markers so the host controller cannot be
# mistaken for a model runtime environment.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

# System Python runs the host-side proxy; Apptainer runs the model worker.
PYTHON_BIN="/usr/bin/python3"
APPTAINER_BIN="/usr/bin/apptainer"

# Inference configuration shared with build_apptainer.sbatch.
CONFIG_FILE="${PROJECT_ROOT}/hpc-job-submit/config.json"
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Cache directories for Hugging Face and PyTorch model weights.
export HF_HOME="$(jq -er '.HF_CACHE_DIR' "${CONFIG_FILE}")"
export TORCH_HOME="$(jq -er '.TORCH_CACHE_DIR' "${CONFIG_FILE}")"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# Persistent Apptainer image produced by build_apptainer.sbatch.
IMAGE_PATH="${IMAGE_DIR}/vhmodels-${MODEL}.sif"

export VHMODELS_IMAGE_PATH="${IMAGE_PATH}"

# Host directories exposed at the same paths inside the container worker.
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}${PROJECT_ROOT}:${PROJECT_ROOT},${HF_HOME}:${HF_HOME},${TORCH_HOME}:${TORCH_HOME}"

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

echo "Running ${MODEL} test.py with ${IMAGE_PATH}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/test.py"
