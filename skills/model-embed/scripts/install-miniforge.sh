#!/usr/bin/env bash
# Install Miniforge (conda + mamba, conda-forge default). No Anaconda defaults channel.
set -euo pipefail

INSTALL_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
BASE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
INSTALLER_PATH="${TMPDIR:-/tmp}/Miniforge3-installer$$"

if command -v conda >/dev/null 2>&1; then
  echo "conda already on PATH: $(conda --version)"
  exit 0
fi

if [[ -x "${INSTALL_PREFIX}/bin/conda" ]]; then
  echo "Miniforge already installed at ${INSTALL_PREFIX}"
  # shellcheck source=/dev/null
  source "${INSTALL_PREFIX}/etc/profile.d/conda.sh"
  conda --version
  exit 0
fi

ensure_curl() {
  if command -v curl >/dev/null 2>&1; then
    return
  fi
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    apt-get install -y curl ca-certificates
    return
  fi
  echo "error: curl is required but not installed" >&2
  exit 1
}

resolve_installer() {
  local os arch linux_arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "${arch}" in
    x86_64 | amd64) arch=x86_64 ;;
    aarch64 | arm64) arch=arm64 ;;
    ppc64le) arch=ppc64le ;;
    *)
      echo "error: unsupported CPU architecture: ${arch}" >&2
      exit 1
      ;;
  esac

  case "${os}" in
    Darwin)
      echo "${BASE_URL}/Miniforge3-Darwin-${arch}.sh"
      ;;
    Linux)
      linux_arch="${arch}"
      [[ "${arch}" == "arm64" ]] && linux_arch=aarch64
      echo "${BASE_URL}/Miniforge3-Linux-${linux_arch}.sh"
      ;;
    MINGW* | MSYS* | CYGWIN*)
      echo "error: native Windows is not supported by this script." >&2
      echo "Download and run: ${BASE_URL}/Miniforge3-Windows-x86_64.exe" >&2
      exit 1
      ;;
    *)
      echo "error: unsupported OS: ${os}" >&2
      exit 1
      ;;
  esac
}

ensure_curl
installer_url="$(resolve_installer)"
echo "Downloading ${installer_url}"
curl -fsSL -o "${INSTALLER_PATH}" "${installer_url}"
bash "${INSTALLER_PATH}" -b -p "${INSTALL_PREFIX}"
rm -f "${INSTALLER_PATH}"

"${INSTALL_PREFIX}/bin/conda" init bash >/dev/null 2>&1 || true
# shellcheck source=/dev/null
source "${INSTALL_PREFIX}/etc/profile.d/conda.sh"
conda config --set channel_priority strict 2>/dev/null || true
conda --version
echo "Miniforge installed at ${INSTALL_PREFIX}"
echo "Run: source ${INSTALL_PREFIX}/etc/profile.d/conda.sh  (or open a new shell after conda init)"
