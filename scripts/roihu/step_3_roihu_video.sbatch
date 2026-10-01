#!/usr/bin/env bash
#SBATCH --job-name=ep24-s3-video
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s3_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s3_%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -euo pipefail
: "${LACLAUGPT_VLLM_TEST_INPUT_CSV:?Set private EP24 input CSV path}"
PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private/ep24_reprocess}
VENV=${LACLAUGPT_VLLM_TEST_VENV:-${PRIVATE_ROOT}/.venv-roihu-vllm}
export LACLAUGPT_MAX_ROWS=${LACLAUGPT_MAX_ROWS:-100}
export LACLAUGPT_VLLM_TEST_OUTPUT_CSV=${LACLAUGPT_VLLM_TEST_OUTPUT_CSV:-${PRIVATE_ROOT}/csv/ep24_video_step_${SLURM_JOB_ID}.csv}
export LACLAUGPT_VLLM_TEST_LOG=${LACLAUGPT_VLLM_TEST_LOG:-${PRIVATE_ROOT}/logs/step_3_video_${SLURM_JOB_ID}.log}
export LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR=${LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR:-${PRIVATE_ROOT}/vllm_video_downloads}
export HF_HOME=${HF_HOME:-${PRIVATE_ROOT}/.hf_cache}
mkdir -p /scratch/project_2009497/logs "${PRIVATE_ROOT}"/{csv,logs,vllm_video_downloads,.hf_cache}
module --force purge
module load python-vllm
if [[ -x "${VENV}/bin/python" ]]; then source "${VENV}/bin/activate"; fi
unset PYTHONPATH
unset PYTHONHOME
cd "${PRIVATE_ROOT}"
srun --ntasks=1 python3 "${PUBLIC_ROOT}/step_3_roihu_video.py"
