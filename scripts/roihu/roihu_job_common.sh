#!/usr/bin/env bash
# Shared runtime setup for EP24 Roihu steps 1-6.
set -euo pipefail

ROIHU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${ROIHU_SCRIPT_DIR}/../.." && pwd)}"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
export LACLAUGPT_PRIVATE_ROOT="${LACLAUGPT_PRIVATE_ROOT:-/scratch/${CSC_PROJECT}/LaclauGPT-Private}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${LACLAUGPT_PRIVATE_ROOT}/analysis/ep24_reprocess}"
export LACLAUGPT_EP24_ENV_FILE="${LACLAUGPT_EP24_ENV_FILE:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.env}"

if [[ -f "${LACLAUGPT_EP24_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LACLAUGPT_EP24_ENV_FILE}"
  set +a
fi

if [[ -n "${LACLAUGPT_EP24_PRIVATE_ROOT:-}" ]]; then
  export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_EP24_PRIVATE_ROOT}"
else
  export LACLAUGPT_EP24_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
fi

export LACLAUGPT_DATASET="${LACLAUGPT_DATASET:-ep2024_reprocess}"
export LACLAUGPT_MONGO_ENABLED="${LACLAUGPT_MONGO_ENABLED:-1}"
export LACLAUGPT_EP24_INPUT_ROOT="${LACLAUGPT_EP24_INPUT_ROOT:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/data/to_reprocess}"
export LACLAUGPT_EP24_OUTPUT_ROOT="${LACLAUGPT_EP24_OUTPUT_ROOT:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/outputs}"
export LACLAUGPT_MULTIMODAL_VENV="${LACLAUGPT_MULTIMODAL_VENV:-${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/.venv-roihu-gpu}"
export LACLAUGPT_VLLM_TEST_VENV="${LACLAUGPT_VLLM_TEST_VENV:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.venv-roihu-vllm}"
export HF_HOME="${HF_HOME:-/scratch/${CSC_PROJECT}/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-/scratch/${CSC_PROJECT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${CSC_PROJECT}/cache}"
export LACLAUGPT_OCR_ENGINE="${LACLAUGPT_OCR_ENGINE:-easyocr}"
export LACLAUGPT_ASR_ENGINE="${LACLAUGPT_ASR_ENGINE:-canary}"

mkdir -p   "/scratch/${CSC_PROJECT}/logs"   "${LACLAUGPT_EP24_OUTPUT_ROOT}"   "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/"{logs,database,Keyframes,Allas}   "${HF_HOME}" "${TORCH_HOME}"

roihu_load_runtime() {
  local runtime="${1:?runtime required: pytorch or vllm}"
  module --force purge
  case "${runtime}" in
    pytorch)
      module load python-pytorch
      module load ffmpeg
      unset PYTHONPATH PYTHONHOME
      [[ -x "${LACLAUGPT_MULTIMODAL_VENV}/bin/python" ]] || {
        echo "Missing Roihu ARM64 venv: ${LACLAUGPT_MULTIMODAL_VENV}" >&2
        echo "Run: bash scripts/roihu/install_roihu.sh" >&2
        exit 2
      }
      # shellcheck disable=SC1091
      source "${LACLAUGPT_MULTIMODAL_VENV}/bin/activate"
      ;;
    vllm)
      module load python-vllm
      module load ffmpeg
      unset PYTHONPATH PYTHONHOME
      if [[ -x "${LACLAUGPT_VLLM_TEST_VENV}/bin/python" ]]; then
        # shellcheck disable=SC1091
        source "${LACLAUGPT_VLLM_TEST_VENV}/bin/activate"
      fi
      ;;
    *) echo "Unknown Roihu runtime: ${runtime}" >&2; exit 2 ;;
  esac
}

roihu_diagnostics() {
  echo "=== EP24 Roihu job diagnostics ==="
  echo "date=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-interactive}"
  echo "arch=$(uname -m)"
  echo "python=$(python3 --version 2>&1)"
  echo "public_root=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}"
  echo "private_root=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
  git -C "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git_sha=/' || true
  nvidia-smi || true
}

roihu_start_ollama() {
  local model="${LACLAUGPT_MULTIMODAL_MODEL:-qwen3.8:27b}"
  local install_root="${OLLAMA_INSTALL_ROOT:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.ollama}"
  export PATH="${install_root}/bin:${PATH}"
  export LD_LIBRARY_PATH="${install_root}/lib/ollama${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.ollama/models}"
  export OLLAMA_HOST="127.0.0.1:$((20000 + ${SLURM_JOB_ID:-1} % 20000))"
  export LACLAUGPT_MULTIMODAL_MODEL="${model}"
  mkdir -p "${OLLAMA_MODELS}"
  command -v ollama >/dev/null 2>&1 || {
    echo "Ollama not installed. Run scripts/roihu/install_roihu.sh or set OLLAMA_INSTALL_ROOT." >&2
    exit 2
  }
  local log="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/logs/ollama_${SLURM_JOB_ID:-interactive}.log"
  ollama serve >"${log}" 2>&1 &
  export ROIHU_OLLAMA_PID=$!
  trap 'kill "${ROIHU_OLLAMA_PID:-}" 2>/dev/null || true' EXIT INT TERM
  for _ in {1..60}; do ollama list >/dev/null 2>&1 && break; sleep 2; done
  ollama list >/dev/null 2>&1 || { echo "Ollama failed; see ${log}" >&2; exit 3; }
  if ! ollama show "${model}" >/dev/null 2>&1; then
    if [[ "${LACLAUGPT_MULTIMODAL_PULL_MODEL:-0}" == "1" ]]; then
      ollama pull "${model}"
    else
      echo "Missing Ollama model: ${model}. Set LACLAUGPT_MULTIMODAL_PULL_MODEL=1 to pull it." >&2
      exit 3
    fi
  fi
}

roihu_preflight() {
  local step="${1:?step required}"
  python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/scripts/roihu/preflight_steps_1_6.py" --step "${step}"
}
