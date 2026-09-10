#!/bin/bash
#SBATCH --job-name=build_apptainer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --acctg-freq=task=1
#SBATCH --time=1:00:00
#SBATCH --output=./output/build_job_output.txt
#SBATCH --error=./output/build_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

# Require submission from the repository root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ ! -f "${PROJECT_ROOT}/vhmodels/envs/Apptainer" ]]; then
    echo "ERROR: Submit this job from the vhmodels repository root." >&2
    exit 1
fi

# Build configuration.
CONFIG_FILE="${PROJECT_ROOT}/hpc-job-submit/config.json"
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
export APPTAINER_CACHEDIR="$(jq -er '.APPTAINER_CACHEDIR' "${CONFIG_FILE}")"
UV_CACHE_DIR="$(jq -er '.UV_CACHE_DIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Check whether the model is registered and retrieve its manifest configuration.
MODEL_CONFIG=""
for candidate in "${PROJECT_ROOT}"/vhmodels/models/*/model.json; do
    if [[ "$(jq -r '.model.id' "${candidate}")" == "${MODEL}" ]]; then
        MODEL_CONFIG="${candidate}"
        break
    fi
done
[[ -n "${MODEL_CONFIG}" ]] || {
    echo "ERROR: Model '${MODEL}' is not registered." >&2
    exit 1
}

MODEL_DIR="$(basename "$(dirname "${MODEL_CONFIG}")")"
PYTHON_VERSION="$(jq -er '.runtimes.apptainer.python' "${MODEL_CONFIG}")"
REQUIREMENTS_FILE="$(jq -er '.runtimes.apptainer.requirements' "${MODEL_CONFIG}")"
TORCH_BACKEND="$(jq -r '.runtimes.apptainer.torch_backend // empty' "${MODEL_CONFIG}")"
EXCLUDE_FILE="$(jq -r '.runtimes.apptainer.exclude // empty' "${MODEL_CONFIG}")"
TORCH_BACKEND_ARG="${TORCH_BACKEND:+--torch-backend ${TORCH_BACKEND}}"
EXCLUDE_ARG="${EXCLUDE_FILE:+--excludes /opt/vhmodels-src/vhmodels/models/${MODEL_DIR}/${EXCLUDE_FILE}}"

# Apptainer executable.
APPTAINER_BIN="/usr/bin/apptainer"

# Final and temporary image paths.
IMAGE_PATH="${IMAGE_DIR}/vhmodels-${MODEL}.sif"
TEMP_IMAGE="${IMAGE_DIR}/.vhmodels-${MODEL}.${SLURM_JOB_ID}.sif.partial"

# User-specific node-local scratch parent.
SCRATCH_PARENT="/localscratch/${USER}"

cd "${PROJECT_ROOT}"

[[ -x "${APPTAINER_BIN}" ]] || {
    echo "ERROR: ${APPTAINER_BIN} is unavailable." >&2
    exit 1
}

"${APPTAINER_BIN}" --version

# Reuse an existing image, if available. Otherwise, build a new one.
if [[ -f "${IMAGE_PATH}" ]]; then
    echo "Apptainer image already exists: ${IMAGE_PATH}"
    exit 0
fi

echo "Building ${IMAGE_PATH} ..."

mkdir -p "${IMAGE_DIR}" \
         "${SCRATCH_PARENT}" \
         "${APPTAINER_CACHEDIR}" \
         "${UV_CACHE_DIR}"

# Keep every temporary build artifact under one unique job directory.
SCRATCH_DIR="$(mktemp -d "${SCRATCH_PARENT}/vhmodels-${MODEL}.${SLURM_JOB_ID}.XXXXXX")"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/apptainer-tmp"
BUILD_CONTEXT="${SCRATCH_DIR}/build-context"

cleanup() {
    rm -rf -- "${SCRATCH_DIR}"
    rm -f -- "${TEMP_IMAGE}"
}
trap cleanup EXIT

mkdir -p "${APPTAINER_TMPDIR}" "${BUILD_CONTEXT}"

cp -a "${PROJECT_ROOT}/vhmodels" "${BUILD_CONTEXT}/vhmodels"
sed \
    -e "s|{project}|${MODEL}|g" \
    -e "s|{model_dir}|${MODEL_DIR}|g" \
    -e "s|{python_version}|${PYTHON_VERSION}|g" \
    -e "s|{requirements_file}|${REQUIREMENTS_FILE}|g" \
    -e "s|{torch_backend_arg}|${TORCH_BACKEND_ARG}|g" \
    -e "s|{exclude_arg}|${EXCLUDE_ARG}|g" \
    "${PROJECT_ROOT}/vhmodels/envs/Apptainer" \
    > "${BUILD_CONTEXT}/Apptainer.def"

(
    cd "${BUILD_CONTEXT}"
    "${APPTAINER_BIN}" build \
        --bind "${UV_CACHE_DIR}:/opt/vhmodels-build-cache" \
        "${TEMP_IMAGE}" Apptainer.def
)

mv -- "${TEMP_IMAGE}" "${IMAGE_PATH}"
trap - EXIT
rm -rf -- "${SCRATCH_DIR}"

echo "Stored Apptainer image at ${IMAGE_PATH}"
