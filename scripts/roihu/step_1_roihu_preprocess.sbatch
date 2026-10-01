#!/bin/bash
#SBATCH --job-name=ep24-s1-preprocess
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s1_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s1_%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL

set -euo pipefail
module load python-pytorch/2.13
export HF_HOME=/scratch/project_2009497/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME"
export LACLAUGPT_MAX_ROWS="${LACLAUGPT_MAX_ROWS:-100}"
export LACLAUGPT_MULTIMODAL_MODEL="${LACLAUGPT_MULTIMODAL_MODEL:-gemma4:12b}"
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
cd "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT"
srun python3 step_1_roihu_preprocess.py
