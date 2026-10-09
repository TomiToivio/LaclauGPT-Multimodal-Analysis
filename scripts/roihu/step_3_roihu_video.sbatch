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

# Slurm executes a copied script from /var/spool/slurmd/job*/slurm_script, so
# BASH_SOURCE[0] does NOT point at this repository inside the job. Resolve the
# checkout from an explicit override, the submit directory, or the standard Roihu
# project path instead.
ROIHU_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-}"
if [[ -z "${ROIHU_PUBLIC_ROOT}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/roihu/roihu_job_common.sh" ]]; then
    ROIHU_PUBLIC_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/roihu_job_common.sh" ]]; then
    ROIHU_PUBLIC_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
  fi
fi
ROIHU_PUBLIC_ROOT="${ROIHU_PUBLIC_ROOT:-/scratch/${CSC_PROJECT:-project_2009497}/LaclauGPT-Multimodal-Analysis}"

if [[ ! -f "${ROIHU_PUBLIC_ROOT}/scripts/roihu/roihu_job_common.sh" ]]; then
  echo "Cannot locate Roihu job helper: ${ROIHU_PUBLIC_ROOT}/scripts/roihu/roihu_job_common.sh" >&2
  echo "Submit from the repository root or set LACLAUGPT_MULTIMODAL_PUBLIC_ROOT explicitly." >&2
  exit 2
fi

export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${ROIHU_PUBLIC_ROOT}"
# shellcheck disable=SC1091
source "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/scripts/roihu/roihu_job_common.sh"

roihu_load_runtime "vllm"
roihu_diagnostics
source "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/scripts/roihu/ensure_step3_rclone.sh"
roihu_ensure_rclone
roihu_preflight 3

cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
source "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/scripts/roihu/lib_roihu_diagnostics.sh"
roihu_diagnostics 3

srun --ntasks=1 python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/ep24_stage_orchestrator.py" \
  --step 3 \
  --script "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/step_3_roihu_video.py" \
  "$@"

echo "EP24 Step 3 completed at $(date --iso-8601=seconds)"
