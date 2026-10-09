#!/usr/bin/env bash
set -euo pipefail
STEP="${1:?usage: setup_step_common.sh STEP}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${CSC_PROJECT}/LaclauGPT-Private/analysis/ep24_reprocess}"
source "${SCRIPT_DIR}/roihu_step_profiles.sh"

NAME="$(roihu_step_name "${STEP}")"
REQ="${PUBLIC_ROOT}/$(roihu_step_requirements "${STEP}")"
VENV="$(roihu_step_venv "${STEP}")"
RUNTIME="$(roihu_step_runtime "${STEP}")"
CACHE_ROOT="/scratch/${CSC_PROJECT}/cache"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export LACLAUGPT_OCR_ENGINE="${LACLAUGPT_OCR_ENGINE:-easyocr}"
export LACLAUGPT_ASR_ENGINE="${LACLAUGPT_ASR_ENGINE:-canary}"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.venvs"   "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/"{logs,database,outputs,Keyframes,Allas}

ARCH="$(uname -m)"
if [[ "${LACLAUGPT_ALLOW_NON_ROIHU_INSTALL:-0}" != "1" ]]; then
  if [[ "${RUNTIME}" == "cpu" ]]; then
    [[ "${ARCH}" == "x86_64" ]] || {
      echo "Step ${STEP} is CPU-only: run setup on roihu-cpu (x86_64); got ${ARCH}." >&2
      exit 2
    }
  else
    [[ "${ARCH}" == "aarch64" ]] || {
      echo "Step ${STEP} requires the Roihu GPU/ARM64 module stack; got ${ARCH}." >&2
      exit 2
    }
  fi
fi
[[ -f "${REQ}" ]] || { echo "Missing requirements: ${REQ}" >&2; exit 2; }

roihu_load_step_modules "${STEP}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  if [[ "${RUNTIME}" == "cpu" ]]; then
    python3 -m venv "${VENV}"
  else
    python3 -m venv --system-site-packages "${VENV}"
  fi
fi
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ}"

if roihu_step_needs_ollama "${STEP}"; then
  OLLAMA_INSTALL_ROOT="${OLLAMA_INSTALL_ROOT:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.ollama}"
  if [[ ! -x "${OLLAMA_INSTALL_ROOT}/bin/ollama" ]]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    mkdir -p "${OLLAMA_INSTALL_ROOT}"
    archive="${tmp}/ollama-linux-arm64.tar.zst"
    curl -fL --retry 3 https://ollama.com/download/ollama-linux-arm64.tar.zst -o "${archive}"
    if tar --help 2>/dev/null | grep -q -- '--zstd'; then
      tar --zstd -xf "${archive}" -C "${OLLAMA_INSTALL_ROOT}"
    elif command -v zstd >/dev/null 2>&1; then
      zstd -dc "${archive}" | tar -xf - -C "${OLLAMA_INSTALL_ROOT}"
    else
      echo "Need tar --zstd or zstd to install Ollama." >&2
      exit 2
    fi
  fi
fi

# Step 3 stages videos through Allas with rclone. CSC module availability can
# differ between login and compute nodes, so install a private ARM64 binary.
if [[ "${STEP}" == "3" ]]; then
  RCLONE_BIN_DIR="${LACLAUGPT_RCLONE_BIN_DIR:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.tools/bin}"
  mkdir -p "${RCLONE_BIN_DIR}"
  if [[ ! -x "${RCLONE_BIN_DIR}/rclone" ]]; then
    if command -v rclone >/dev/null 2>&1; then
      ln -sfn "$(command -v rclone)" "${RCLONE_BIN_DIR}/rclone"
    else
      tmp_rclone="$(mktemp -d)"
      trap 'rm -rf "${tmp_rclone}"' EXIT
      curl -fL --retry 3 https://downloads.rclone.org/rclone-current-linux-arm64.zip -o "${tmp_rclone}/rclone.zip"
      python - "${tmp_rclone}/rclone.zip" "${RCLONE_BIN_DIR}/rclone" <<'PY'
import pathlib, sys, zipfile
archive, destination = map(pathlib.Path, sys.argv[1:])
with zipfile.ZipFile(archive) as zf:
    matches = [entry for entry in zf.infolist() if entry.filename.endswith('/rclone') and not entry.is_dir()]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one rclone executable in download")
    destination.write_bytes(zf.read(matches[0]))
destination.chmod(0o755)
PY
      rm -rf "${tmp_rclone}"
      trap - EXIT
    fi
  fi
  export PATH="${RCLONE_BIN_DIR}:${PATH}"
  rclone version | head -1
fi

export LACLAUGPT_STEP="${STEP}"
export LACLAUGPT_STEP_VENV="${VENV}"
python "${SCRIPT_DIR}/validate_step_environment.py" --step "${STEP}" --setup

echo "EP24 Step ${STEP} (${NAME}) setup complete"
echo "runtime=${RUNTIME}"
echo "venv=${VENV}"
echo "requirements=${REQ}"
