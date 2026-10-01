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
PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private/ep24_reprocess}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu-gpu}
export LACLAUGPT_MAX_ROWS=${LACLAUGPT_MAX_ROWS:-100}
export HF_HOME=${HF_HOME:-/scratch/project_2009497/hf-cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}

mkdir -p /scratch/project_2009497/logs "${PRIVATE_ROOT}"/{csv,Allas,Keyframes,database,logs,whisper}
module --force purge
module load python-pytorch
module load ffmpeg
unset PYTHONPATH
unset PYTHONHOME
[[ -x "${VENV}/bin/python" ]] || { echo "Missing Roihu ARM64 venv: ${VENV}" >&2; exit 2; }
source "${VENV}/bin/activate"
cd "${PRIVATE_ROOT}"
srun --ntasks=1 python3 "${PUBLIC_ROOT}/ep24_stage_orchestrator.py" --step 1 --script "${PUBLIC_ROOT}/step_1_roihu_preprocess.py"
