#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"

# The GPU login node is ARM64, while Step 9 runs on x86_64 CPU nodes.
# Build its venv on the same architecture as the eventual Slurm job.
if [[ "$(uname -m)" == "x86_64" ]]; then
  exec "${SCRIPT_DIR}/setup_step_common.sh" 9
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Unsupported Step 9 setup architecture: $(uname -m)" >&2
  exit 2
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "Step 9 setup on ARM64 requires Slurm sbatch to install on a CPU node." >&2
  exit 2
}

LOG_DIR="/scratch/${CSC_PROJECT}/logs"
mkdir -p "${LOG_DIR}"
echo "Submitting x86_64 Step 9 environment setup from the GPU terminal."
# Wait for completion and propagate failures. No ARM64 wheels are installed
# into the CPU environment even when launched from roihu-gpu.
sbatch --wait --account="${CSC_PROJECT}" --partition=small \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --job-name=ep24-step9-setup \
  --output="${LOG_DIR}/ep24_step9_setup_%j.out" \
  --error="${LOG_DIR}/ep24_step9_setup_%j.err" \
  --export=ALL,LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${PUBLIC_ROOT}" \
  --wrap="bash ${SCRIPT_DIR}/setup_step_common.sh 9"
echo "Step 9 CPU environment setup finished. Logs: ${LOG_DIR}/ep24_step9_setup_*.out"
