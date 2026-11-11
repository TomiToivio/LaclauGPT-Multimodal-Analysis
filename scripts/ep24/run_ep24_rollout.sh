#!/usr/bin/env bash
# #373: explicit read-only planning unless --execute and sample approval are provided.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:?Usage: run_ep24_rollout.sh continue|fresh [--execute] [--approval-file PATH] [--input-root PATH]}"
shift
[[ "${MODE}" == "continue" || "${MODE}" == "fresh" ]] || { echo "Expected continue|fresh" >&2; exit 2; }
EXECUTE=0
PRIVATE_ROOT="${LACLAUGPT_EP24_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess}"
# Private, persistent operator sign-off. Never generate approval automatically.
APPROVAL="${LACLAUGPT_EP24_APPROVAL_FILE:-${PRIVATE_ROOT}/ep24_sample_approved.txt}"
INPUT_ROOT="${LACLAUGPT_EP24_INPUT_ROOT:-/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess}"
while (($#)); do
  case "$1" in
    --execute|--submit) EXECUTE=1; shift ;;
    --approval-file) APPROVAL="${2:?approval file missing}"; shift 2 ;;
    --input-root) INPUT_ROOT="${2:?input root missing}"; shift 2 ;;
    --dry-run) shift ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done
ARGS=(--mode "${MODE}" --input-root "${INPUT_ROOT}")
if (( EXECUTE )); then
  [[ -f "${APPROVAL}" && -s "${APPROVAL}" ]] || {
    echo "Sample approval file missing/empty: ${APPROVAL}" >&2
    echo "Review the dry-run and create a signed approval file before --execute." >&2
    exit 2
  }
  echo "Using sample approval: ${APPROVAL}" >&2
  [[ "${LACLAUGPT_DATASET:-ep2024_reprocess}" == "ep2024_reprocess" ]] || {
    echo "Verify and explicitly configure the correct sample namespace before executing" >&2; exit 2;
  }
  command -v sbatch >/dev/null || { echo "sbatch unavailable" >&2; exit 2; }
  ARGS+=(--apply --approval-file "${APPROVAL}")
fi
# The Python planner verifies existing sample matches or refuses to modify MongoDB.
python3 "${ROOT}/scripts/ep24/ep24_rollout.py" "${ARGS[@]}"
if (( ! EXECUTE )); then
  echo "Read-only plan complete; no MongoDB changes or Slurm jobs submitted."
  exit 0
fi
mapfile -t COUNTRIES < <(python3 - "${INPUT_ROOT}" "${MODE}" "${ROOT}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from ep24_bootstrap import ordered_country_files, country_slug
for path in ordered_country_files(Path(sys.argv[1])):
    name = country_slug(path)
    if (name in {"finland", "poland", "portugal"}) == (sys.argv[2] == "continue"):
        print(name)
PY
)
for country in "${COUNTRIES[@]}"; do
  echo "Submitting stages 1-7 for ${country}; existing completed stages remain complete"
  bash "${ROOT}/scripts/roihu/run_country_pipeline.sh" "${country}" --limit 0 --from-step 1 --to-step 7
done
echo "Submission complete. Inspect Slurm jobs, stages, manifests, and off-node backup verification."
