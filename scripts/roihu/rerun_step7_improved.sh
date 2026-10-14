#!/usr/bin/env bash
# Rerun improved EP24 Step 7 on CSC Roihu. Run on login node.
set -euo pipefail
ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
COUNTRY="${1:-finland}"
MODE="${2:-preview}"
case "$COUNTRY" in finland|poland|portugal|germany|spain|hungary|croatia|france|bulgaria|sweden) ;; *) echo "Unsupported country $COUNTRY" >&2; exit 2;; esac
cd "$ROOT"
git pull --ff-only
export LACLAUGPT_COUNTRIES="$COUNTRY"
export LACLAUGPT_COUNTRY="$COUNTRY"
export LACLAUGPT_STEP7_MAX_CONTEXT_CHARS="${LACLAUGPT_STEP7_MAX_CONTEXT_CHARS:-28000}"
export LACLAUGPT_STEP7_NUM_PREDICT="${LACLAUGPT_STEP7_NUM_PREDICT:-4096}"
printf 'Country: %s | mode: %s | revision: %s\n' "$COUNTRY" "$MODE" "$(git rev-parse --short HEAD)"
case "$MODE" in
  preview)
    # Read-only eligibility check (will only count pending/retry/error).
    bash scripts/roihu/run_step.sh 7 --country "$COUNTRY" --retry-errors --dry-run
    ;;
  retry)
    # Re-run failed or not-yet-started records; completed rows remain untouched.
    bash scripts/roihu/run_step.sh 7 --country "$COUNTRY" --retry-errors
    ;;
  full)
    # --force resets ALL Stage-7 completion markers for this country.
    # This mutates Mongo and replaces Step-7 outputs; archive externally first.
    [[ "${STEP7_ALLOW_FORCE:-}" == "YES" ]] || {
      echo "Refusing full rerun without STEP7_ALLOW_FORCE=YES. Archive your old results first." >&2
      exit 2
    }
    bash scripts/roihu/run_step.sh 7 --country "$COUNTRY" --force
    ;;
  *) echo "Usage: bash scripts/roihu/rerun_step7_improved.sh COUNTRY [preview|retry|full]" >&2; exit 2 ;;
esac
