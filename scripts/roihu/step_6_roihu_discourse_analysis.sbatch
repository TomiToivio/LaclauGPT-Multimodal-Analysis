#!/usr/bin/env bash
#SBATCH --job-name=ep24-s6-discourse
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s6_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s6_%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/roihu_job_common.sh"

roihu_load_runtime "pytorch"
roihu_diagnostics
roihu_start_ollama
roihu_preflight 6

cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
source "${PUBLIC_ROOT}/scripts/roihu/lib_roihu_diagnostics.sh"
roihu_diagnostics 6

srun --ntasks=1 python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/ep24_stage_orchestrator.py" \
  --step 6 \
  --script "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/step_6_roihu_discourse_analysis.py" \
  "$@"

echo "EP24 Step 6 completed at $(date --iso-8601=seconds)"
