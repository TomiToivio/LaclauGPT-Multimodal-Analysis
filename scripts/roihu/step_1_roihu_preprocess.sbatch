#!/usr/bin/env bash
#SBATCH --job-name=ep24-s1-preprocess
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s1_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s1_%j.err
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
roihu_preflight 1

cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
srun --ntasks=1 python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/ep24_stage_orchestrator.py" \
  --step 1 \
  --script "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/step_1_roihu_preprocess.py" \
  "$@"

echo "EP24 Step 1 completed at $(date --iso-8601=seconds)"
