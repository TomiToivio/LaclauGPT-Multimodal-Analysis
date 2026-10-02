#!/usr/bin/env bash
#SBATCH --job-name=ep24-s3-video
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s3_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s3_%j.err
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

roihu_load_runtime "vllm"
roihu_diagnostics
roihu_preflight 3

cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
srun --ntasks=1 python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/ep24_stage_orchestrator.py" \
  --step 3 \
  --script "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/step_3_roihu_video.py" \
  "$@"

echo "EP24 Step 3 completed at $(date --iso-8601=seconds)"
