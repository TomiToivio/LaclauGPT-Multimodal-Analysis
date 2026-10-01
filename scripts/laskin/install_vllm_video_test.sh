#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${LACLAUGPT_LASKIN_VENV:-${PUBLIC_ROOT}/.venv-laskin-vllm}"
PYTHON_BIN="${LACLAUGPT_LASKIN_PYTHON:-python3.10}"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: ${PYTHON_BIN} is required" >&2; exit 2; }
"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${PUBLIC_ROOT}/experiments/requirements-vllm-video-test-laskin.txt"
"${VENV}/bin/python" - <<'PY'
import torch, transformers, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
PY