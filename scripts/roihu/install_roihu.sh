#!/usr/bin/env bash
# Bootstrap EP24 steps 1-6 on Roihu-GPU (NVIDIA Grace/GH200).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
export LACLAUGPT_PRIVATE_ROOT="${LACLAUGPT_PRIVATE_ROOT:-/scratch/${CSC_PROJECT}/LaclauGPT-Private}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${LACLAUGPT_PRIVATE_ROOT}/analysis/ep24_reprocess}"
ENV_FILE="${LACLAUGPT_EP24_ENV_FILE:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.env}"
VENV="${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu-gpu}"

[[ -f "${PUBLIC_ROOT}/step_1_roihu_preprocess.py" ]] || { echo "Run this from the LaclauGPT-Multimodal-Analysis checkout." >&2; exit 2; }

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -n "${LACLAUGPT_EP24_PRIVATE_ROOT:-}" ]]; then
  export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_EP24_PRIVATE_ROOT}"
else
  export LACLAUGPT_EP24_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
fi

[[ -d "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}" ]] || { echo "Missing private EP24 root: ${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}" >&2; exit 2; }

if [[ "$(uname -m)" != "aarch64" && "${LACLAUGPT_ALLOW_NON_ROIHU_INSTALL:-0}" != "1" ]]; then
  echo "Expected Roihu-GPU ARM64 login node (roihu-gpu.csc.fi); got $(uname -m)." >&2
  exit 2
fi

module --force purge
module load python-pytorch
module load ffmpeg
unset PYTHONPATH PYTHONHOME

mkdir -p   "/scratch/${CSC_PROJECT}/logs"   "/scratch/${CSC_PROJECT}/cache/"{huggingface,torch,pip}   "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/"{logs,database,outputs,Keyframes,Allas,.ollama/models}

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/${CSC_PROJECT}/cache/pip}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PUBLIC_ROOT}/requirements-roihu-steps1-6.txt"

OLLAMA_INSTALL_ROOT="${OLLAMA_INSTALL_ROOT:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.ollama}"
if [[ ! -x "${OLLAMA_INSTALL_ROOT}/bin/ollama" ]]; then
  echo "Installing Ollama ARM64 locally under ${OLLAMA_INSTALL_ROOT}"
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' EXIT
  mkdir -p "${OLLAMA_INSTALL_ROOT}"
  curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz -o "${tmp}/ollama.tgz"
  tar -xzf "${tmp}/ollama.tgz" -C "${OLLAMA_INSTALL_ROOT}"
fi

export PATH="${OLLAMA_INSTALL_ROOT}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/${CSC_PROJECT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/scratch/${CSC_PROJECT}/cache/torch}"
export LACLAUGPT_OCR_ENGINE="${LACLAUGPT_OCR_ENGINE:-easyocr}"
export LACLAUGPT_ASR_ENGINE="${LACLAUGPT_ASR_ENGINE:-canary}"

python - <<'PY'
import importlib, platform
mods = ["pandas", "pymongo", "redis", "cv2", "ollama", "pydantic", "easyocr", "nemo.collections.asr"]
missing=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f"{m}: {type(e).__name__}: {e}")
print("architecture:", platform.machine())
if missing:
    raise SystemExit("Import validation failed:\n  " + "\n  ".join(missing))
print("EP24 steps 1-6 Python imports: OK")
PY

echo
echo "Roihu bootstrap complete."
echo "venv=${VENV}"
echo "private_root=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
echo "ollama=$(command -v ollama)"
echo "Next: configure ${ENV_FILE}, then submit a Finland --limit 1 smoke job."
