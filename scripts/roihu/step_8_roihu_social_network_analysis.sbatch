#!/usr/bin/env bash
#SBATCH --job-name=ep24-s8-sna
#SBATCH --account=project_2009497
#SBATCH --output=/scratch/project_2009497/logs/ep24_s8_%j.out
#SBATCH --error=/scratch/project_2009497/logs/ep24_s8_%j.err
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
python3 "${PUBLIC_ROOT}/scripts/roihu/check_ep24_runtime.py"
cd "${PRIVATE_ROOT}"
MODEL=${LACLAUGPT_MULTIMODAL_MODEL:-gemma4:12b}
OLLAMA_INSTALL_ROOT=${OLLAMA_INSTALL_ROOT:-${PRIVATE_ROOT}/.ollama}
export PATH="${OLLAMA_INSTALL_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${OLLAMA_INSTALL_ROOT}/lib/ollama${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OLLAMA_MODELS=${OLLAMA_MODELS:-${PRIVATE_ROOT}/.ollama/models}
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export LACLAUGPT_MULTIMODAL_MODEL="${MODEL}"
mkdir -p "${PRIVATE_ROOT}/.ollama/models"
command -v ollama >/dev/null 2>&1 || { echo "Ollama missing from ${OLLAMA_INSTALL_ROOT}" >&2; exit 2; }
OLLAMA_LOG="${PRIVATE_ROOT}/logs/ollama_${SLURM_JOB_ID}.log"
ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT INT TERM
for _ in {1..60}; do ollama list >/dev/null 2>&1 && break; sleep 2; done
ollama list >/dev/null 2>&1 || { echo "Ollama failed; see ${OLLAMA_LOG}" >&2; exit 3; }
if ! ollama show "${MODEL}" >/dev/null 2>&1; then
  [[ ${LACLAUGPT_MULTIMODAL_PULL_MODEL:-0} == 1 ]] && ollama pull "${MODEL}" || { echo "Missing model: ${MODEL}" >&2; exit 3; }
fi
srun --ntasks=1 python3 "${PUBLIC_ROOT}/ep24_stage_orchestrator.py" --step 8 --script "${PUBLIC_ROOT}/step_8_roihu_social_network_analysis.py" "$@"
