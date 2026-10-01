#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "${PUBLIC_ROOT}" ]]; then
  PUBLIC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}
STAGES=${LACLAUGPT_MULTIMODAL_STAGES:-"preprocess frame summary postprocess populism"}

mkdir -p "${PRIVATE_ROOT}"/{csv,Allas,Keyframes,database,logs,whisper}
cd "${PRIVATE_ROOT}"

run_stage() {
  local stage=$1
  local script="${PUBLIC_ROOT}/roihu_${stage}.py"
  [[ -f "${script}" ]] || { echo "Missing stage script: ${script}" >&2; exit 2; }
  echo "=== LaclauGPT multimodal stage ${stage} ==="
  echo "script=${script}"
  echo "model=${LACLAUGPT_MULTIMODAL_MODEL:-unset}"
  python "${script}"
}

for stage in ${STAGES}; do
  if [[ "${stage}" == "populism" ]] \
     && [[ "${LACLAUGPT_ENRICHMENT_ENABLED:-0}" =~ ^(1|true|yes|on)$ ]] \
     && [[ " ${STAGES} " != *" enrich "* ]]; then
    run_stage enrich
  fi
  case "${stage}" in
    preprocess|frame|summary|postprocess|enrich|populism) run_stage "${stage}" ;;
    *) echo "Unknown stage: ${stage}" >&2; exit 2 ;;
  esac
done
