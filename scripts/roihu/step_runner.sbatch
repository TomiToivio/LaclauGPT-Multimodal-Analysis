#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

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

# The Step 3 setup installs a project-local ARM64 rclone binary. Restore its
# location in the Slurm job after the CSC module stack is loaded.
if [[ "${STEP}" == "3" ]]; then
  export PATH="${LACLAUGPT_RCLONE_BIN_DIR:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.tools/bin}:${PATH}"
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
