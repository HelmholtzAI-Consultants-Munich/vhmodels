#!/bin/bash
#SBATCH --job-name=build_apptainer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --acctg-freq=task=1
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

CONFIG_FILE="${1:?Usage: sbatch build_apptainer.sbatch /absolute/path/config.json}"
if [[ "${CONFIG_FILE}" != /* || ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Pass an absolute path to an existing config.json." >&2
    exit 1
fi
PYTHON_BIN="${VHMODELS_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"

# Apptainer caches OCI layers; the package CLI binds the uv cache into %post.
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
export APPTAINER_CACHEDIR="$(jq -er '.APPTAINER_CACHEDIR' "${CONFIG_FILE}")"
export UV_CACHE_DIR="$(jq -er '.UV_CACHE_DIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"

# Apptainer executable.
APPTAINER_BIN="/usr/bin/apptainer"

# Final and temporary image paths.
IMAGE_PATH="${IMAGE_DIR}/vhmodels-${MODEL}.sif"
TEMP_IMAGE="${IMAGE_DIR}/.vhmodels-${MODEL}.${SLURM_JOB_ID}.partial.sif"

# User-specific node-local scratch parent.
SCRATCH_PARENT="/localscratch/${USER}"

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
export TMPDIR="${SCRATCH_DIR}"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/apptainer-tmp"
export PROOT_TMP_DIR="${SCRATCH_DIR}/proot-tmp"

cleanup() {
    rm -rf -- "${SCRATCH_DIR}"
    rm -f -- "${TEMP_IMAGE}"
}
trap cleanup EXIT

mkdir -p "${APPTAINER_TMPDIR}" "${PROOT_TMP_DIR}"

"${PYTHON_BIN}" -m vhmodels.vh_checker.cli create-apptainer-image \
    "${MODEL}" --output "${TEMP_IMAGE}"

mv -- "${TEMP_IMAGE}" "${IMAGE_PATH}"
trap - EXIT
rm -rf -- "${SCRATCH_DIR}"

echo "Stored Apptainer image at ${IMAGE_PATH}"
