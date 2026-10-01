#!/usr/bin/env bash
#SBATCH --job-name=ep24-vllm-video-test
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --output=vllm-video-test-%j.out
#SBATCH --error=vllm-video-test-%j.err

# Standalone smoke test: native whole-video analysis with vLLM/Qwen3-VL-8B on
# CSC Roihu-GPU. Deliberately NOT part of the production pipeline: it does not
# call run_pipeline.sh and does not write to the pipeline CSVs.
#
# One full GH200 (96 GB) is ample for an 8B model.
#
# Submit from a directory you are happy to hold the Slurm logs (private project
# storage is the safe choice, because the log mentions dataset paths):
#
#   sbatch scripts/roihu/vllm_video_test.sbatch

set -euo pipefail

# ---- required configuration (never hard-coded, never committed) -----------
: "${LACLAUGPT_VLLM_TEST_INPUT_CSV:?Set the private EP24 input CSV path}"
: "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:?Set the public repository checkout path}"
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set the private runtime root}"

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}
VENV=${LACLAUGPT_VLLM_TEST_VENV:-${PRIVATE_ROOT}/.venv-roihu-vllm}
MODEL=${LACLAUGPT_VLLM_TEST_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

# Working locations: private scratch, not HOME.
DOWNLOAD_DIR=${LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR:-${PRIVATE_ROOT}/vllm_video_downloads}
LOG_PATH=${LACLAUGPT_VLLM_TEST_LOG:-${PRIVATE_ROOT}/logs/vllm_video_test_${SLURM_JOB_ID}.log}
OUTPUT_CSV=${LACLAUGPT_VLLM_TEST_OUTPUT_CSV:-${PRIVATE_ROOT}/csv/ep24_vllm_video_test_${SLURM_JOB_ID}.csv}
export HF_HOME=${HF_HOME:-${PRIVATE_ROOT}/.hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}

mkdir -p "${PRIVATE_ROOT}"/{logs,csv,vllm_video_downloads,"${HF_HOME##*/}"}

[[ -d "${PUBLIC_ROOT}" ]] || { echo "Missing public checkout: ${PUBLIC_ROOT}" >&2; exit 2; }
[[ -f "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" ]] || {
  echo "Missing input CSV: ${LACLAUGPT_VLLM_TEST_INPUT_CSV}" >&2; exit 2; }

# ---- CSC module environment ----------------------------------------------
# vLLM is a separate module on Roihu-GPU (python-vllm). Do not also load
# python-pytorch: the vLLM module already brings its own PyTorch build.
module --force purge
module load python-vllm

# Roihu is aarch64; never reuse a Puhti x86 virtualenv here.
if [[ -x "${VENV}/bin/python" ]]; then
  source "${VENV}/bin/activate"
else
  echo "NOTE: no project venv at ${VENV}; using the module python directly." >&2
  echo "      If extra packages are needed, create it as documented in" >&2
  echo "      docs/VLLM_VIDEO_TEST.md and re-submit." >&2
fi

# Keep the job off any user site-packages.
unset PYTHONPATH
unset PYTHONHOME

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname) ==="
echo "python   : $(command -v python)"
python --version || true
echo "input    : ${LACLAUGPT_VLLM_TEST_INPUT_CSV}"
echo "output   : ${OUTPUT_CSV}"
echo "log      : ${LOG_PATH}"

# ---- optional Allas plumbing ----------------------------------------------
# Allas is S3-compatible. Configure an rclone remote in the job environment
# (e.g. via RCLONE_CONFIG pointing at a private rclone.conf) and pass the bucket
# through ALLAS_BUCKET. Credentials stay in that private config: they are never
# in this repository and never on the command line.
#
# To stage the videos without Allas, mount/copy them under ALLAS_LOCAL_ROOT and
# run with --fetch-backend local.
if ! command -v rclone >/dev/null 2>&1; then
  echo "NOTE: rclone not on PATH. Use --fetch-backend local with a staged mirror," >&2
  echo "      or load the Allas tooling documented for Roihu." >&2
fi

# ---- run ------------------------------------------------------------------
cd "${PRIVATE_ROOT}"
set -x
python "${PUBLIC_ROOT}/experiments/vllm_video_test.py" \
  --input-csv "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --log-path "${LOG_PATH}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --model "${MODEL}" \
  --sample-size "${LACLAUGPT_VLLM_TEST_SAMPLE_SIZE:-5}" \
  --seed "${LACLAUGPT_VLLM_TEST_SEED:-20261001}" \
  --fetch-backend "${LACLAUGPT_VLLM_TEST_FETCH_BACKEND:-rclone}" \
  "$@"
set +x

echo
echo "=== done ==="
echo "result CSV : ${OUTPUT_CSV}"
echo "debug log  : ${LOG_PATH}"
echo "slurm out  : slurm-${SLURM_JOB_ID:-?}.out (or as configured)"
