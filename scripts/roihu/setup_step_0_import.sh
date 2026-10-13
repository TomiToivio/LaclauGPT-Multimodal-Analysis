#!/usr/bin/env bash
# Step 0 is CPU-only; run on Roihu login or CPU node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIVATE="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${CSC_PROJECT:-project_2009497}/LaclauGPT-Private/analysis/ep24_reprocess}"
VENV="${LACLAUGPT_STEP0_VENV:-${PRIVATE}/.venvs/step-0}"
mkdir -p "$(dirname "${VENV}")"
if [[ ! -x "${VENV}/bin/python" ]]; then python3 -m venv "${VENV}"; fi
"${VENV}/bin/python" -m pip install 'pandas>=2,<4' 'pymongo>=4,<5'
echo "Step 0 ready: ${VENV}/bin/python"
