#!/usr/bin/env bash
#SBATCH --job-name=ep24-issue128-bench
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_issue128_bench_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_issue128_bench_%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=12:00:00

set -euo pipefail
PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu-gpu}
: "${LACLAUGPT_BENCH_MANIFEST:?Set LACLAUGPT_BENCH_MANIFEST to a PRIVATE FI/PL/PT benchmark manifest}"
export LACLAUGPT_BENCH_OUTPUT=${LACLAUGPT_BENCH_OUTPUT:-/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/benchmarks/issue128}
export HF_HOME=${HF_HOME:-/scratch/project_2009497/hf-cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}

mkdir -p /scratch/project_2009497/logs "${LACLAUGPT_BENCH_OUTPUT}"
module --force purge
module load python-pytorch
module load gcc/14.3.0 ffmpeg
unset PYTHONPATH
unset PYTHONHOME
[[ -x "${VENV}/bin/python" ]] || { echo "Missing Roihu ARM64 venv: ${VENV}" >&2; exit 2; }
source "${VENV}/bin/activate"

python3 "${PUBLIC_ROOT}/scripts/roihu/check_ep24_runtime.py"
srun --ntasks=1 python3 "${PUBLIC_ROOT}/scripts/roihu/benchmark_issue128_backends.py"
