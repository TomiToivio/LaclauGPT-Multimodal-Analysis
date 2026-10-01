#!/usr/bin/env bash
#SBATCH --job-name=ep24-vllm-10video
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=212G
#SBATCH --time=02:00:00
#SBATCH --output=vllm-video-test-%j.out
#SBATCH --error=vllm-video-test-%j.err

# Issue #24: isolated EP24 native-video vLLM experiment on CSC Roihu.
#
# Submit the allocation on the command line because #SBATCH directives do not
# expand ordinary shell environment variables:
#
#   sbatch --account="$CSC_PROJECT" scripts/roihu/vllm_video_test.sbatch
#
# This script intentionally does NOT start Ollama. CSC's python-vllm module is
# the inference runtime. One Roihu GH200 is requested, matching current CSC
# gpumedium syntax.

set -euo pipefail

# Public checkout path is optional. Prefer an explicit override; otherwise
# resolve the Git checkout from the directory where sbatch was submitted.
# Slurm copies this script to /var/spool/slurmd, so BASH_SOURCE points at the
# spool copy and cannot be used to discover the repository checkout.
if [[ -n "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-}" ]]; then
  PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}"
else
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
  if command -v git >/dev/null 2>&1 && git -C "${SUBMIT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
    PUBLIC_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel)"
  else
    PUBLIC_ROOT="${SUBMIT_DIR}"
  fi
fi

: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set private runtime root}"
: "${LACLAUGPT_VLLM_TEST_INPUT_CSV:?Set prepared private ~10-video CSV}"
# ALLAS_BUCKET is optional for issue #24 because the prepared EP24 sample normally
# carries full HTTP(S) allas_filename URLs. It is only required for rows that
# contain a bucket-relative object key instead of a full URL.
ALLAS_BUCKET="${ALLAS_BUCKET:-}"

PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
VENV="${LACLAUGPT_VLLM_TEST_VENV:-${PRIVATE_ROOT}/.venv-roihu-vllm}"
MODEL="${LACLAUGPT_VLLM_TEST_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
RCLONE_REMOTE="${RCLONE_REMOTE:-s3allas}"

OUTPUT_DIR="${LACLAUGPT_VLLM_TEST_OUTPUT_DIR:-${PRIVATE_ROOT}/csv}"
LOG_DIR="${LACLAUGPT_VLLM_TEST_LOG_DIR:-${PRIVATE_ROOT}/logs}"
DOWNLOAD_DIR="${LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR:-${PRIVATE_ROOT}/vllm_video_downloads}"
OUTPUT_CSV="${LACLAUGPT_VLLM_TEST_OUTPUT_CSV:-${OUTPUT_DIR}/ep24_vllm_video_test_${SLURM_JOB_ID}.csv}"
LOG_PATH="${LACLAUGPT_VLLM_TEST_LOG:-${LOG_DIR}/vllm_video_test_${SLURM_JOB_ID}.log}"

export HF_HOME="${HF_HOME:-${PRIVATE_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${DOWNLOAD_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

[[ -f "${PUBLIC_ROOT}/experiments/vllm_video_test.py" ]] || {
  echo "ERROR: experiment script missing under detected public repo root: ${PUBLIC_ROOT}" >&2
  echo "Submit from the LaclauGPT-Multimodal-Analysis checkout, or set" >&2
  echo "LACLAUGPT_MULTIMODAL_PUBLIC_ROOT=/path/to/LaclauGPT-Multimodal-Analysis" >&2
  exit 2
}
[[ -f "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" ]] || {
  echo "ERROR: input CSV missing: ${LACLAUGPT_VLLM_TEST_INPUT_CSV}" >&2; exit 2; }
[[ -x "${VENV}/bin/python" ]] || {
  echo "ERROR: Roihu vLLM venv missing: ${VENV}" >&2
  echo "Create it using docs/VLLM_VIDEO_TEST.md before submitting." >&2
  exit 2
}

module --force purge
module load python-vllm
module load allas
module load ffmpeg
source "${VENV}/bin/activate"

unset PYTHONPATH
unset PYTHONHOME

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found" >&2; exit 2; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe not found" >&2; exit 2; }
command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not found; module load allas failed?" >&2; exit 2; }

echo "============================================================"
echo "EP24 vLLM 10-video experiment"
echo "job id       : ${SLURM_JOB_ID:-<unset>}"
echo "host         : $(hostname)"
echo "public repo  : ${PUBLIC_ROOT}"
echo "input CSV    : ${LACLAUGPT_VLLM_TEST_INPUT_CSV}"
echo "output CSV   : ${OUTPUT_CSV}"
echo "python log   : ${LOG_PATH}"
echo "work dir     : ${DOWNLOAD_DIR}"
echo "model        : ${MODEL}"
if [[ -n "${ALLAS_BUCKET}" ]]; then
  echo "Allas remote : ${RCLONE_REMOTE}:<configured bucket>"
else
  echo "Allas remote : full per-row allas_filename URL (ALLAS_BUCKET not required)"
fi
echo "initial skip : 1.0 s (mandatory)"
echo "============================================================"

python - <<'PY'
import importlib
for name in ("vllm", "torch", "transformers", "qwen_vl_utils", "pandas", "av"):
    m = importlib.import_module(name)
    print(f"{name:14s} {getattr(m, '__version__', '<unknown>')}")
PY

# Check that the configured Allas remote is usable without printing credentials.
rclone lsd "${RCLONE_REMOTE}:" >/dev/null

cd "${PUBLIC_ROOT}"

EXTRA_ARGS=()
if [[ "${LACLAUGPT_VLLM_TEST_KEEP_DOWNLOADS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--keep-downloads)
fi

# The private CSV is already the reproducible ~10-video sample. Do not select a
# second random sample in the batch wrapper. Passing sample-size equal to the
# row count preserves compatibility with the current harness until its CLI uses
# an explicit --all-rows mode.
ROW_COUNT=$(python - "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
print(len(df))
PY
)

if [[ "${ROW_COUNT}" -lt 1 ]]; then
  echo "ERROR: input CSV is empty" >&2
  exit 2
fi

echo "prepared sample rows: ${ROW_COUNT}"

srun python experiments/vllm_video_test.py \
  --input-csv "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --log-path "${LOG_PATH}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --model "${MODEL}" \
  --sample-size "${ROW_COUNT}" \
  --seed 20261001 \
  --fetch-backend rclone \
  --rclone-remote "${RCLONE_REMOTE}" \
  --allas-bucket "${ALLAS_BUCKET}" \
  --gpu-memory-utilization "${LACLAUGPT_VLLM_TEST_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${LACLAUGPT_VLLM_TEST_MAX_MODEL_LEN:-32768}" \
  --max-tokens "${LACLAUGPT_VLLM_TEST_MAX_TOKENS:-2048}" \
  --temperature "${LACLAUGPT_VLLM_TEST_TEMPERATURE:-0.1}" \
  --top-p "${LACLAUGPT_VLLM_TEST_TOP_P:-0.9}" \
  --video-min-pixels "${LACLAUGPT_VLLM_TEST_VIDEO_MIN_PIXELS:-4096}" \
  --video-max-pixels "${LACLAUGPT_VLLM_TEST_VIDEO_MAX_PIXELS:-262144}" \
  --video-total-pixels "${LACLAUGPT_VLLM_TEST_VIDEO_TOTAL_PIXELS:-20971520}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "DONE: ${OUTPUT_CSV}"
echo "DEBUG LOG: ${LOG_PATH}"
