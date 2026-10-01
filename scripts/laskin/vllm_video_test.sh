#!/usr/bin/env bash
# Run the Laskin vLLM video experiment (issue #32).
#
#   LACLAUGPT_VLLM_TEST_INPUT_CSV=/path/to/sample.csv \
#   LASKIN_VLLM_VENV=$HOME/.venvs/laskin-vllm-video \
#   bash scripts/laskin/vllm_video_test.sh
#
# Optional overrides (all environment variables, no secrets in this file):
#   LACLAUGPT_VLLM_TEST_OUTPUT_CSV   result CSV path
#   LACLAUGPT_VLLM_TEST_LOG          debug log path
#   LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR local media staging dir
#   LACLAUGPT_VLLM_TEST_MODEL        model id (default is the Laskin-viable default)
#   LACLAUGPT_VLLM_TEST_SEED         selection seed
#   LACLAUGPT_VLLM_TEST_VIDEO_API    auto | modern | legacy | mm_processor_kwargs | direct
#   LACLAUGPT_VLLM_TEST_FETCH_BACKEND rclone | local | none
#   LASKIN_VLLM_HF_HOME              model cache (kept outside HOME by default)
#
# This script does not assume its working directory and does not modify the
# machine's system CUDA/PyTorch installation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="${REPO_ROOT}/experiments/vllm_video_test.py"

VENV="${LASKIN_VLLM_VENV:-$HOME/.venvs/laskin-vllm-video}"
WORK_DIR="${LACLAUGPT_VLLM_TEST_WORKDIR:-$HOME/laskin-vllm-video}"

# Laskin/Volta cannot run Qwen3-VL (see docs/LASKIN_VLLM_VIDEO_TEST.md), so the
# default here is the closest sibling that the pinned 0.8.x stack supports.
MODEL="${LACLAUGPT_VLLM_TEST_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
INPUT_CSV="${LACLAUGPT_VLLM_TEST_INPUT_CSV:-}"
OUTPUT_CSV="${LACLAUGPT_VLLM_TEST_OUTPUT_CSV:-${WORK_DIR}/ep24_vllm_video_test_laskin.csv}"
LOG_PATH="${LACLAUGPT_VLLM_TEST_LOG:-${WORK_DIR}/logs/vllm_video_test_laskin.log}"
DOWNLOAD_DIR="${LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR:-${WORK_DIR}/downloads}"
HF_HOME="${LASKIN_VLLM_HF_HOME:-${WORK_DIR}/hf}"

echo "=== Laskin vLLM video experiment ==="
echo "repo        : ${REPO_ROOT}"
echo "harness     : ${HARNESS}"
echo "venv        : ${VENV}"
echo "model       : ${MODEL}"
echo "input_csv   : ${INPUT_CSV:-<unset>}"
echo "output_csv  : ${OUTPUT_CSV}"
echo "log         : ${LOG_PATH}"
echo "downloads   : ${DOWNLOAD_DIR}"
echo "hf_home     : ${HF_HOME}"
echo "hostname    : $(hostname)"
echo "git_head    : $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"
echo

# --- fail early on anything required ---------------------------------------
[[ -f "${HARNESS}" ]] || { echo "Missing harness: ${HARNESS}" >&2; exit 2; }
[[ -n "${INPUT_CSV}" ]] || {
  echo "Set LACLAUGPT_VLLM_TEST_INPUT_CSV to the prepared sample CSV." >&2; exit 2; }
[[ -f "${INPUT_CSV}" ]] || { echo "Input CSV not found: ${INPUT_CSV}" >&2; exit 2; }
[[ -x "${VENV}/bin/python" ]] || {
  echo "Missing venv at ${VENV}." >&2
  echo "Create it first: bash ${REPO_ROOT}/scripts/laskin/install_vllm_video_test.sh" >&2
  exit 2; }

mkdir -p "${WORK_DIR}/logs" "${DOWNLOAD_DIR}" "${HF_HOME}" "$(dirname "${OUTPUT_CSV}")"

export HF_HOME
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Report the two facts that decide whether this can work at all.
"${VENV}/bin/python" - <<'PY'
import torch
print("torch         :", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("arch_list     :", torch.cuda.get_arch_list())
print("sm_70_in_build:", any("70" in a for a in torch.cuda.get_arch_list()))
PY
echo

CMD=(
  "${VENV}/bin/python" "${HARNESS}"
  --input-csv "${INPUT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --log-path "${LOG_PATH}"
  --download-dir "${DOWNLOAD_DIR}"
  --model "${MODEL}"
  --video-api "${LACLAUGPT_VLLM_TEST_VIDEO_API:-auto}"
  --fetch-backend "${LACLAUGPT_VLLM_TEST_FETCH_BACKEND:-rclone}"
  --seed "${LACLAUGPT_VLLM_TEST_SEED:-20261001}"
)
# The prepared private CSV is the experiment sample. Process every row by
# default, as issue #24 requires. Sampling is opt-in only.
if [[ -n "${LACLAUGPT_VLLM_TEST_SAMPLE_SIZE:-}" ]]; then
  CMD+=(--sample-size "${LACLAUGPT_VLLM_TEST_SAMPLE_SIZE}")
fi

set -x
exec "${CMD[@]}"
