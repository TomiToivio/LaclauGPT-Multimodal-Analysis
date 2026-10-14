#!/usr/bin/env bash
# Step 9 RDF exporter, on CSC Roihu login node (CPU Slurm job).
set -euo pipefail
ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
COUNTRY="${1:-finland}"
MODE="${2:-preview}"
case "$COUNTRY" in finland|poland|portugal|germany|spain|hungary|croatia|france|bulgaria|sweden) ;; *) echo "Invalid country: $COUNTRY" >&2; exit 2;; esac
cd "$ROOT"
git pull --ff-only
export LACLAUGPT_COUNTRIES="$COUNTRY" LACLAUGPT_COUNTRY="$COUNTRY"
export LACLAUGPT_DATASET="${LACLAUGPT_DATASET:-ep2024_reprocess}"
echo "Step 9 country=$COUNTRY mode=$MODE revision=$(git rev-parse --short HEAD)"
case "$MODE" in
 preview)
   bash scripts/roihu/run_step.sh 9 --country "$COUNTRY" --dry-run ;;
 run)
   # Process only Step-8 complete records that have no Step-9 completion.
   bash scripts/roihu/run_step.sh 9 --country "$COUNTRY" ;;
 retry)
   bash scripts/roihu/run_step.sh 9 --country "$COUNTRY" --retry-errors ;;
 full)
   # Danger: force resets Step-9 markers. It does not reset Step 8.
   [[ "${STEP9_ALLOW_FORCE:-}" == YES ]] || { echo "Archive old RDF output and set STEP9_ALLOW_FORCE=YES first" >&2; exit 2; }
   bash scripts/roihu/run_step.sh 9 --country "$COUNTRY" --force ;;
 *)
   echo "Usage: bash scripts/roihu/run_step9_improved.sh COUNTRY [preview|run|retry|full]" >&2; exit 2 ;;
esac
