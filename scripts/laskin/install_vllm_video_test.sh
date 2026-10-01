#!/usr/bin/env bash
# Create the isolated Laskin vLLM video-test environment (issue #32).
#
#   bash scripts/laskin/install_vllm_video_test.sh [target-dir]
#
# Default target: ~/.venvs/laskin-vllm-video
#
# Design choices, deliberately:
#   * A dedicated venv WITHOUT --system-site-packages. The machine has a broken
#     vLLM under ~/.local (module import fails) and a cu130 torch that cannot
#     initialise against the 12.8 driver; inheriting system site-packages lets
#     either shadow the working stack.
#   * The machine's CUDA/PyTorch installation is never modified.
#   * Versions are pinned in experiments/requirements-vllm-video-test-laskin.txt
#     and the reason for each pin is documented there.
#
# This script only creates an environment. It downloads no models and touches no
# private data.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS="${REPO_ROOT}/experiments/requirements-vllm-video-test-laskin.txt"
TARGET="${1:-$HOME/.venvs/laskin-vllm-video}"
PYTHON_BIN="${LASKIN_VLLM_PYTHON:-python3.10}"

echo "=== Laskin vLLM video-test environment ==="
echo "repo        : ${REPO_ROOT}"
echo "requirements: ${REQUIREMENTS}"
echo "target venv : ${TARGET}"
echo "python      : ${PYTHON_BIN}"
echo

[[ -f "${REQUIREMENTS}" ]] || { echo "Missing requirements file: ${REQUIREMENTS}" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Interpreter '${PYTHON_BIN}' not found. Set LASKIN_VLLM_PYTHON to a 3.10/3.11 python." >&2
  exit 2
}

# Report the constraint that decides everything, before building.
echo "--- GPU check ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  echo "  NOTE: vLLM wheels need compute capability >= 7.5."
  echo "        Volta (7.0) uses the pinned 0.8.x stack in this file."
else
  echo "  nvidia-smi not found: this machine has no visible NVIDIA GPU." >&2
fi
echo

if [[ -x "${TARGET}/bin/python" ]]; then
  echo "venv already exists: ${TARGET}"
  echo "re-run with a different directory, or remove it first to rebuild."
else
  echo "--- creating venv (no system site-packages) ---"
  "${PYTHON_BIN}" -m venv "${TARGET}"
fi

echo "--- installing pinned stack (this downloads ~3 GB) ---"
"${TARGET}/bin/pip" install --upgrade pip
"${TARGET}/bin/pip" install -r "${REQUIREMENTS}"

echo
echo "--- verifying the stack ---"
"${TARGET}/bin/python" - <<'PY'
import sys
print("python        :", sys.version.split()[0])
try:
    import torch
    print("torch         :", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("arch_list     :", torch.cuda.get_arch_list())
    print("sm_70_in_build:", any("70" in a for a in torch.cuda.get_arch_list()))
except Exception as exc:
    print("torch probe failed:", type(exc).__name__, exc)
try:
    import vllm
    print("vllm          :", vllm.__version__)
except Exception as exc:
    print("vllm import failed:", type(exc).__name__, exc)
try:
    import transformers, qwen_vl_utils, av
    print("transformers  :", transformers.__version__)
    print("qwen_vl_utils : installed")
    print("av            :", av.__version__)
except Exception as exc:
    print("support import failed:", type(exc).__name__, exc)
PY

cat <<EOF

=== done ===
Use it with:
  LASKIN_VLLM_VENV=${TARGET} bash ${REPO_ROOT}/scripts/laskin/vllm_video_test.sh
EOF
