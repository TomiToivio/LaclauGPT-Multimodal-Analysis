#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
export PYTHONUNBUFFERED=1
export LACLAUGPT_PRINT_ANALYSIS="${LACLAUGPT_PRINT_ANALYSIS:-1}"
echo "EP24 job started: step=${LACLAUGPT_STEP:-unknown} job=${SLURM_JOB_ID:-unknown} $(date --iso-8601=seconds)"
trap 'status=$?; echo "EP24 job exit: step=${LACLAUGPT_STEP:-unknown} code=$status at $(date --iso-8601=seconds)" >&2' EXIT

STEP="${LACLAUGPT_STEP:?LACLAUGPT_STEP is required}"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/${CSC_PROJECT:-project_2009497}/LaclauGPT-Multimodal-Analysis}"
SCRIPT_DIR="${PUBLIC_ROOT}/scripts/roihu"
source "${SCRIPT_DIR}/roihu_step_profiles.sh"
ENTRY="${PUBLIC_ROOT}/$(roihu_step_entrypoint "${STEP}")"
VENV="${LACLAUGPT_STEP_VENV:-$(roihu_step_venv "${STEP}")}"

roihu_load_step_modules "${STEP}"
[[ -x "${VENV}/bin/python" ]] || { echo "Missing Step ${STEP} venv: ${VENV}" >&2; exit 2; }
source "${VENV}/bin/activate"
export LACLAUGPT_MULTIMODAL_VENV="${VENV}"
[[ "${STEP}" == "3" ]] && export LACLAUGPT_VLLM_TEST_VENV="${VENV}"

if [[ -f "${LACLAUGPT_EP24_ENV_FILE:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.env}" ]]; then
  set -a
  source "${LACLAUGPT_EP24_ENV_FILE:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.env}"
  set +a
fi

export LACLAUGPT_DATASET="${LACLAUGPT_DATASET:-ep2024_reprocess}"
echo "EP24 Mongo namespace: dataset=${LACLAUGPT_DATASET} database=${LACLAUGPT_MONGO_DATABASE:-<URI/default>}"
export HF_HOME="${HF_HOME:-/scratch/${CSC_PROJECT:-project_2009497}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/scratch/${CSC_PROJECT:-project_2009497}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${CSC_PROJECT:-project_2009497}/cache}"
export LACLAUGPT_OCR_ENGINE="${LACLAUGPT_OCR_ENGINE:-easyocr}"
export LACLAUGPT_ASR_ENGINE="${LACLAUGPT_ASR_ENGINE:-canary}"
mkdir -p "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"/{logs,database,outputs,Keyframes,Allas} "${HF_HOME}" "${TORCH_HOME}"

if roihu_step_needs_ollama "${STEP}"; then
  source "${SCRIPT_DIR}/roihu_job_common.sh"
  roihu_start_ollama
fi

# Recover a missing Step 3 rclone before the fail-fast environment check.
if [[ "${STEP}" == "3" ]]; then
  source "${SCRIPT_DIR}/ensure_step3_rclone.sh"
  roihu_ensure_rclone
  export RCLONE_BIN="$(command -v rclone)"
  [[ -x "${RCLONE_BIN}" ]] || { echo "Step 3 rclone binary missing" >&2; exit 2; }
  echo "step3_rclone_executable=${RCLONE_BIN}"
fi

python "${SCRIPT_DIR}/validate_step_environment.py" --step "${STEP}"

echo "run_started=$(date --iso-8601=seconds)"
echo "step=${STEP}"
echo "entrypoint=${ENTRY}"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "git_sha=$(git -C "${PUBLIC_ROOT}" rev-parse HEAD 2>/dev/null || true)"
echo "python=$(command -v python)"
echo "input_root=${LACLAUGPT_EP24_INPUT_ROOT:-default}"
echo "output_root=${LACLAUGPT_EP24_OUTPUT_ROOT:-default}"
echo "model=${LACLAUGPT_MULTIMODAL_MODEL:-default}"

cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
if [[ "${LACLAUGPT_MONGO_ENABLED:-0}" =~ ^(1|true|yes|on)$ ]]; then
  srun --ntasks=1 python "${PUBLIC_ROOT}/ep24_stage_orchestrator.py" --step "${STEP}" --script "${ENTRY}" "$@"
else
  srun --ntasks=1 python "${ENTRY}" "$@"
  python "${PUBLIC_ROOT}/ep24_result_reporting.py" --step "${STEP}" "$@"
fi
echo "run_completed=$(date --iso-8601=seconds)"
