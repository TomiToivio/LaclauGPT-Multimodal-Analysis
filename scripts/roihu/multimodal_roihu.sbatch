#!/usr/bin/env bash
#SBATCH --job-name=laclaugpt-multimodal
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00

set -euo pipefail

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu-gpu}
MODEL=${LACLAUGPT_MULTIMODAL_MODEL:-qwen3.8:27b}
OLLAMA_INSTALL_ROOT=${OLLAMA_INSTALL_ROOT:-${PRIVATE_ROOT}/.ollama}

[[ -d "${PUBLIC_ROOT}" ]] || { echo "Missing public checkout: ${PUBLIC_ROOT}" >&2; exit 2; }
mkdir -p "${PRIVATE_ROOT}"/{csv,Allas,Keyframes,database,logs,whisper,.ollama/models}

module --force purge
module load python-pytorch
module load gcc/14.3.0 ffmpeg
unset PYTHONPATH
unset PYTHONHOME

[[ -x "${VENV}/bin/python" ]] || {
  echo "Missing Roihu ARM64 venv: ${VENV}" >&2
  echo "Create it on roihu-gpu.csc.fi; do not reuse a Puhti x86 venv." >&2
  exit 2
}
source "${VENV}/bin/activate"

export PATH="${OLLAMA_INSTALL_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${OLLAMA_INSTALL_ROOT}/lib/ollama${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OLLAMA_MODELS=${OLLAMA_MODELS:-${PRIVATE_ROOT}/.ollama/models}
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export LACLAUGPT_MULTIMODAL_MODEL="${MODEL}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

command -v ollama >/dev/null 2>&1 || {
  echo "Ollama is not installed in ${OLLAMA_INSTALL_ROOT} or PATH." >&2
  echo "Install the ARM64 Ollama build in private/project storage before the job." >&2
  exit 2
}

OLLAMA_LOG="${PRIVATE_ROOT}/logs/ollama_${SLURM_JOB_ID}.log"
ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT INT TERM

for _ in {1..60}; do
  ollama list >/dev/null 2>&1 && break
  sleep 2
done
ollama list >/dev/null 2>&1 || {
  echo "Ollama failed to start; see ${OLLAMA_LOG}" >&2
  exit 3
}

if ! ollama show "${MODEL}" >/dev/null 2>&1; then
  if [[ ${LACLAUGPT_MULTIMODAL_PULL_MODEL:-0} == 1 ]]; then
    command -v flock >/dev/null 2>&1 || {
      echo "flock is required for a shared Ollama model store." >&2
      exit 3
    }
    exec {pull_fd}>"${OLLAMA_MODELS}/.laclaugpt-pull.lock"
    if ! flock -w "${LACLAUGPT_OLLAMA_PULL_LOCK_SECONDS:-1800}" "${pull_fd}"; then
      echo "Timed out waiting for shared Ollama model-store lock." >&2
      exit 3
    fi
    if ! ollama show "${MODEL}" >/dev/null 2>&1; then
      PULL_LOG="${PRIVATE_ROOT}/logs/ollama_pull_${SLURM_JOB_ID:-interactive}.log"
      pull_ok=0
      for attempt in 1 2; do
        echo "Pulling ${MODEL} (attempt ${attempt}/2); log=${PULL_LOG}" >&2
        if ollama pull "${MODEL}" >>"${PULL_LOG}" 2>&1 &&
           ollama show "${MODEL}" >/dev/null 2>&1; then
          pull_ok=1
          break
        fi
        tail -n 12 "${PULL_LOG}" >&2 || true
        if [[ ${attempt} -lt 2 ]]; then sleep 5; fi
      done
      if [[ "${pull_ok}" != 1 ]]; then
        echo "Ollama model pull failed after two attempts; see ${PULL_LOG}." >&2
        exit 3
      fi
    fi
    flock -u "${pull_fd}"
    exec {pull_fd}>&-
  else
    echo "Required model missing: ${MODEL}" >&2
    echo "Pull it before submission or set LACLAUGPT_MULTIMODAL_PULL_MODEL=1." >&2
    exit 3
  fi
fi

cd "${PRIVATE_ROOT}"
srun --ntasks=1 bash "${PUBLIC_ROOT}/scripts/roihu/run_pipeline.sh"
