#!/usr/bin/env bash
# Execute on CSC Roihu login node; Step 8 only, never Step 1-7.
set -euo pipefail
ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
COUNTRY="${1:-finland}"
MODE="${2:-preview}"
case "$COUNTRY" in finland|poland|portugal|germany|spain|hungary|croatia|france|bulgaria|sweden) ;; *) echo "Unsupported country: $COUNTRY" >&2; exit 2;; esac
cd "$ROOT"
git pull --ff-only
export LACLAUGPT_COUNTRIES="$COUNTRY" LACLAUGPT_COUNTRY="$COUNTRY"
export LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS="${LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS:-16000}"
echo "Step 8 country=$COUNTRY mode=$MODE git=$(git rev-parse --short HEAD)"
case "$MODE" in
 preview)
   bash scripts/roihu/run_step.sh 8 --country "$COUNTRY" --retry-errors --dry-run ;;
 retry)
   # Reruns failed/pending records only. Prior completed old-version rows stay complete.
   bash scripts/roihu/run_step.sh 8 --country "$COUNTRY" --retry-errors ;;
 full)
   # WARNING: replaces Step 8 data. Archive Mongo and CSV before running.
   [[ "${STEP8_ALLOW_FORCE:-}" == YES ]] || { echo "Set STEP8_ALLOW_FORCE=YES after archiving current Step 8 results." >&2; exit 2; }
   bash scripts/roihu/run_step.sh 8 --country "$COUNTRY" --force ;;
 *)
   echo "Usage: bash scripts/roihu/retry_step8_improved.sh COUNTRY [preview|retry|full]" >&2; exit 2 ;;
esac
