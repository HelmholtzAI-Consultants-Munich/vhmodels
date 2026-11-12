#!/bin/bash
#SBATCH --job-name=build_ray_env
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

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch ray/build_env.sbatch /absolute/path/config.json <MODEL>|--all [--force]" >&2
    exit 1
fi
CONFIG_FILE="$1"
if [[ "${CONFIG_FILE}" != /* || ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Pass an absolute path to an existing config.json." >&2
    exit 1
fi
# Must be the Ray driver Python: the venv gets its Python and Ray versions.
PYTHON_BIN="${VHMODELS_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not in PATH. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/0.12.3/install.sh | sh" >&2
    exit 1
}

# The uv cache is shared with build_image.sbatch, so wheels download once.
export UV_CACHE_DIR="$(jq -er '.UV_CACHE_DIR' "${CONFIG_FILE}")"
export VHMODELS_VENV_DIR="$(jq -er '.VENV_DIR' "${CONFIG_FILE}")"

uv --version
"${PYTHON_BIN}" -m vhmodels.vh_checker.cli create-ray-env "${@:2}"
