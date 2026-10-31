#!/usr/bin/env bash
#SBATCH --job-name=ep24-s9-rdf
#SBATCH --account=project_2009497
#SBATCH --output=logs/ep24_s9_%j.out
#SBATCH --error=logs/ep24_s9_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${LACLAUGPT_EP24_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess}}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${PRIVATE_ROOT}"
export LACLAUGPT_EP24_PRIVATE_ROOT="${PRIVATE_ROOT}"

# Use precisely the x86_64 CPU virtualenv installed by setup_step_9_rdf.sh.
source "${PUBLIC_ROOT}/scripts/roihu/roihu_step_profiles.sh"
roihu_load_step_modules 9
VENV="$(roihu_step_venv 9)"
mkdir -p /scratch/project_2009497/logs "${PRIVATE_ROOT}/rdf" "${PRIVATE_ROOT}/database"
[[ "$(uname -m)" == "x86_64" ]] || { echo "Step 9 requires x86_64 CPU; got $(uname -m)" >&2; exit 2; }
[[ -x "${VENV}/bin/python" ]] || {
  echo "Missing Step 9 CPU venv: ${VENV}; run bash scripts/roihu/setup_step_9_rdf.sh first." >&2
  exit 2
}
source "${VENV}/bin/activate"
export LACLAUGPT_STEP_VENV="${VENV}"
export LACLAUGPT_STEP=9
python3 "${PUBLIC_ROOT}/scripts/roihu/validate_step_environment.py" --step 9
python3 "${PUBLIC_ROOT}/scripts/roihu/check_ep24_runtime.py"
echo "step=9 runtime=cpu arch=$(uname -m) python=$(python3 --version 2>&1) git_sha=$(git -C "${PUBLIC_ROOT}" rev-parse --short HEAD)"
echo "private_root=${PRIVATE_ROOT} country=${LACLAUGPT_COUNTRY:-default} limit=${LACLAUGPT_MAX_ROWS:-0}"
cd "${PRIVATE_ROOT}"
srun --ntasks=1 python3 "${PUBLIC_ROOT}/ep24_stage_orchestrator.py" \
  --step 9 --script "${PUBLIC_ROOT}/step_9_roihu_rdf.py" "$@"
