#!/usr/bin/env bash
set -euo pipefail
STEP="${1:?usage: run_step.sh STEP [sbatch args...]}"
shift
PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}
PRIVATE_REPO=${LACLAUGPT_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private}
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${PRIVATE_REPO}/analysis/ep24_reprocess}
ENV_FILE=${LACLAUGPT_EP24_ENV_FILE:-${PRIVATE_ROOT}/.env}

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export LACLAUGPT_MONGO_ENABLED=${LACLAUGPT_MONGO_ENABLED:-1}
export LACLAUGPT_DATASET=${LACLAUGPT_DATASET:-ep2024_reprocess}
export LACLAUGPT_EP24_INPUT_ROOT=${LACLAUGPT_EP24_INPUT_ROOT:-${PRIVATE_ROOT}/data/to_reprocess}
export LACLAUGPT_EP24_OUTPUT_ROOT=${LACLAUGPT_EP24_OUTPUT_ROOT:-${PRIVATE_ROOT}/outputs}
mkdir -p /scratch/project_2009497/logs "${LACLAUGPT_EP24_OUTPUT_ROOT}"

SBATCH="${PUBLIC_ROOT}/scripts/roihu/step_${STEP}_roihu_"
case "${STEP}" in
  1) SBATCH+="preprocess.sbatch" ;;
  2) SBATCH+="frame.sbatch" ;;
  3) SBATCH+="video.sbatch" ;;
  4) SBATCH+="summary.sbatch" ;;
  5) SBATCH+="postprocess.sbatch" ;;
  6) SBATCH+="discourse_analysis.sbatch" ;;
  7) SBATCH+="discourse_network_analysis.sbatch" ;;
  8) SBATCH+="social_network_analysis.sbatch" ;;
  9) SBATCH+="rdf.sbatch" ;;
  *) echo "Invalid step: ${STEP}" >&2; exit 2 ;;
esac
[[ -f "${SBATCH}" ]] || { echo "Missing sbatch: ${SBATCH}" >&2; exit 2; }
command -v sbatch >/dev/null || { echo "sbatch not found; run this on CSC Roihu login node" >&2; exit 2; }
echo "Submitting EP24 Step ${STEP}: ${SBATCH}"
sbatch --export=ALL "${SBATCH}" "$@"
