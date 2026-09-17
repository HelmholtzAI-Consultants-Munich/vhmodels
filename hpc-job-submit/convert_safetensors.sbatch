#!/bin/bash
#SBATCH --job-name=convert_safetensors
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=./output/convert_safetensors_job_output.txt
#SBATCH --error=./output/convert_safetensors_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

# Require submission from the repository root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ ! -f "${PROJECT_ROOT}/hpc-job-submit/config.json" ]]; then
    echo "ERROR: Submit this job from the vhmodels repository root." >&2
    exit 1
fi

# Override these at submission time to convert another repository or revision.
MODEL_ID="${MODEL_ID:-virtual-human-chc/prot_t5_xxl_bfd}"
SOURCE_REVISION="${SOURCE_REVISION:-main}"

CONFIG_FILE="${PROJECT_ROOT}/hpc-job-submit/config.json"
IMAGE_DIR="$(jq -er '.IMAGE_DIR' "${CONFIG_FILE}")"
IMAGE_PATH="${IMAGE_DIR}/vhmodels-prottrans.sif"
APPTAINER_BIN="/usr/bin/apptainer"

# Pin the official Hugging Face converter for reproducibility.
CONVERTER_REPO="safetensors/convert"
CONVERTER_REVISION="37a6a610ddc70577a6c55f26097e0af3c44659a7"

SCRATCH_PARENT="/localscratch/${USER}"
REQUIRED_SCRATCH_KB=$((120 * 1024 * 1024))

[[ -x "${APPTAINER_BIN}" ]] || {
    echo "ERROR: ${APPTAINER_BIN} is unavailable." >&2
    exit 1
}
[[ -f "${IMAGE_PATH}" ]] || {
    echo "ERROR: ${IMAGE_PATH} does not exist. Build the prottrans image first." >&2
    exit 1
}

mkdir -p "${SCRATCH_PARENT}"

AVAILABLE_SCRATCH_KB="$(df -Pk "${SCRATCH_PARENT}" | awk 'NR == 2 {print $4}')"
if (( AVAILABLE_SCRATCH_KB < REQUIRED_SCRATCH_KB )); then
    echo "ERROR: Safetensors conversion requires at least 120 GiB free under ${SCRATCH_PARENT}." >&2
    df -h "${SCRATCH_PARENT}" >&2
    exit 1
fi

SCRATCH_DIR="$(mktemp -d "${SCRATCH_PARENT}/safetensors-conversion.${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="${SCRATCH_DIR}/tmp"
CONVERTER_DIR="${SCRATCH_DIR}/converter"

cleanup() {
    rm -rf -- "${SCRATCH_DIR}"
}
trap cleanup EXIT

mkdir -p "${TMPDIR}" "${CONVERTER_DIR}"

echo "Converting ${MODEL_ID} at revision ${SOURCE_REVISION}"
echo "Temporary storage: ${SCRATCH_DIR}"
df -h "${SCRATCH_PARENT}"

# Keep authentication in the normal Hugging Face configuration under $HOME.
unset HF_HOME

# Download the pinned official converter into node-local scratch.
"${APPTAINER_BIN}" exec \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    --env "TMPDIR=${TMPDIR}" \
    "${IMAGE_PATH}" \
    /opt/venv/bin/python -c \
    'from huggingface_hub import hf_hub_download; import sys; hf_hub_download(repo_id=sys.argv[1], filename="convert.py", repo_type="space", revision=sys.argv[2], local_dir=sys.argv[3])' \
    "${CONVERTER_REPO}" "${CONVERTER_REVISION}" "${CONVERTER_DIR}"

# Fail before downloading the checkpoint if no usable Hugging Face login exists.
"${APPTAINER_BIN}" exec \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    --env "TMPDIR=${TMPDIR}" \
    "${IMAGE_PATH}" \
    /opt/venv/bin/python -c \
    'from huggingface_hub import HfApi; identity = HfApi().whoami(); print("Authenticated with Hugging Face as {}".format(identity["name"]))'

# The official converter validates tensor equality and opens the Hub PR.
"${APPTAINER_BIN}" exec \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    --env "TMPDIR=${TMPDIR}" \
    "${IMAGE_PATH}" \
    /opt/venv/bin/python "${CONVERTER_DIR}/convert.py" \
    "${MODEL_ID}" --revision "${SOURCE_REVISION}" -y
