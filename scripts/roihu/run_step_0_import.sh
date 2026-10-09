#!/usr/bin/env bash
# Import private CSV -> Mongo, with an auditable private log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIVATE="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${CSC_PROJECT:-project_2009497}/LaclauGPT-Private/analysis/ep24_reprocess}"
VENV="${LACLAUGPT_STEP0_VENV:-${PRIVATE}/.venvs/step-0}"
[[ -x "${VENV}/bin/python" ]] || {
  echo "Step 0 environment missing. Run bash scripts/roihu/setup_step_0_import.sh" >&2
  exit 2
}
LOG_DIR="${PRIVATE}/logs/step0"
mkdir -p "${LOG_DIR}"
chmod 700 "${LOG_DIR}"
LOG="${LOG_DIR}/import_$(date +%Y%m%dT%H%M%S)_$$.log"
export PYTHONUNBUFFERED=1
echo "Step 0 log: ${LOG}" >&2
set -o pipefail
"${VENV}/bin/python" -u "${ROOT}/step_0_roihu_import.py" "$@" 2>&1 | tee "${LOG}"
