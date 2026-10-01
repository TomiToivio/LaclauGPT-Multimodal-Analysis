#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${LACLAUGPT_LASKIN_VENV:-${PUBLIC_ROOT}/.venv-laskin-vllm}"
: "${LACLAUGPT_VLLM_TEST_INPUT_CSV:?Set the private prepared EP24 input CSV path}"
[[ -f "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" ]] || { echo "ERROR: input CSV not found" >&2; exit 2; }
[[ -x "${VENV}/bin/python" ]] || { echo "ERROR: run install_vllm_video_test.sh first" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg is required" >&2; exit 2; }
command -v ffprobe >/dev/null || { echo "ERROR: ffprobe is required" >&2; exit 2; }

WORK_ROOT="${LACLAUGPT_VLLM_TEST_WORK_ROOT:-${TMPDIR:-/tmp}/laclaugpt-vllm-video-test}"
OUTPUT_DIR="${LACLAUGPT_VLLM_TEST_OUTPUT_DIR:-${WORK_ROOT}/results}"
LOG_DIR="${LACLAUGPT_VLLM_TEST_LOG_DIR:-${WORK_ROOT}/logs}"
DOWNLOAD_DIR="${LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR:-${WORK_ROOT}/downloads}"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${DOWNLOAD_DIR}" "${HF_HOME}"
export HF_HOME

MODEL="${LACLAUGPT_VLLM_TEST_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_CSV="${LACLAUGPT_VLLM_TEST_OUTPUT_CSV:-${OUTPUT_DIR}/vllm_video_test_laskin.csv}"
LOG_PATH="${LACLAUGPT_VLLM_TEST_LOG:-${LOG_DIR}/vllm_video_test_laskin.log}"
FETCH_BACKEND="${LACLAUGPT_VLLM_TEST_FETCH_BACKEND:-rclone}"

echo "EP24 Laskin vLLM experiment"
echo "public_root=${PUBLIC_ROOT}"
echo "input_csv=${LACLAUGPT_VLLM_TEST_INPUT_CSV}"
echo "output_csv=${OUTPUT_CSV}"
echo "model=${MODEL}"
echo "video_api=direct"

exec "${VENV}/bin/python" "${PUBLIC_ROOT}/experiments/vllm_video_test.py" \
  --input-csv "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --log-path "${LOG_PATH}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --fetch-backend "${FETCH_BACKEND}" \
  --model "${MODEL}" \
  --video-api direct \
  --max-model-len "${LACLAUGPT_VLLM_TEST_MAX_MODEL_LEN:-16384}" \
  --max-tokens "${LACLAUGPT_VLLM_TEST_MAX_TOKENS:-2048}" \
  --gpu-memory-utilization "${LACLAUGPT_VLLM_TEST_GPU_MEMORY_UTILIZATION:-0.85}" \
  ${LACLAUGPT_VLLM_TEST_KEEP_DOWNLOADS:+--keep-downloads}