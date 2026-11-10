#!/bin/bash
#SBATCH --job-name=pull_image
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

CONFIG_FILE="${1:?Usage: sbatch pull_image.sbatch /absolute/path/config.json}"
if [[ "${CONFIG_FILE}" != /* || ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Pass an absolute path to an existing config.json." >&2
    exit 1
fi

IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
export APPTAINER_CACHEDIR="$(jq -er '.APPTAINER_CACHEDIR' "${CONFIG_FILE}")"
MODEL="$(jq -er '.MODEL' "${CONFIG_FILE}")"
IMAGE_TAG="$(jq -r '.IMAGE_TAG // "latest"' "${CONFIG_FILE}")"

# OCI image on GHCR, built by .github/workflows/images.yaml.
IMAGE_REPO="${VHMODELS_IMAGE_REPO:-ghcr.io/helmholtzai-consultants-munich}"
IMAGE_REF="${IMAGE_REPO}/vhmodels-${MODEL}:${IMAGE_TAG}"

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

mkdir -p "${IMAGE_DIR}" "${SCRATCH_PARENT}" "${APPTAINER_CACHEDIR}"

# Keep every temporary conversion artifact under one unique job directory.
SCRATCH_DIR="$(mktemp -d "${SCRATCH_PARENT}/vhmodels-${MODEL}.${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="${SCRATCH_DIR}"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/apptainer-tmp"

cleanup() {
    rm -rf -- "${SCRATCH_DIR}"
    rm -f -- "${TEMP_IMAGE}"
}
trap cleanup EXIT

mkdir -p "${APPTAINER_TMPDIR}"

# The OCI image plus the worker %startscript of the Apptainer runtime.
DEFINITION="${SCRATCH_DIR}/vhmodels-${MODEL}.def"
cat > "${DEFINITION}" <<DEF
Bootstrap: docker
From: ${IMAGE_REF}

%startscript
    exec /opt/venv/bin/python -m vhmodels.vh_checker.worker serve "\$@"
DEF

echo "Converting ${IMAGE_REF} ..."
"${APPTAINER_BIN}" build "${TEMP_IMAGE}" "${DEFINITION}"

mv -f -- "${TEMP_IMAGE}" "${IMAGE_PATH}"
trap - EXIT
rm -rf -- "${SCRATCH_DIR}"

echo "Stored Apptainer image at ${IMAGE_PATH}"
