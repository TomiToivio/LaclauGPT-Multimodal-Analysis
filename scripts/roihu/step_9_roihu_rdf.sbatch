#!/usr/bin/env bash
#SBATCH --job-name=ep24-s9-rdf
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s9_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s9_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -euo pipefail
PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private/ep24_reprocess}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu-gpu}
export LACLAUGPT_MAX_ROWS=${LACLAUGPT_MAX_ROWS:-100}
mkdir -p /scratch/project_2009497/logs "${PRIVATE_ROOT}/rdf" "${PRIVATE_ROOT}/database"
[[ -x "${VENV}/bin/python" ]] || { echo "Missing Roihu venv: ${VENV}" >&2; exit 2; }
source "${VENV}/bin/activate"
python3 "${PUBLIC_ROOT}/scripts/roihu/check_ep24_runtime.py"
cd "${PRIVATE_ROOT}"
srun --ntasks=1 python3 "${PUBLIC_ROOT}/ep24_stage_orchestrator.py" --step 9 --script "${PUBLIC_ROOT}/step_9_roihu_rdf.py" "$@"
