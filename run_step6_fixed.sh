#!/usr/bin/env bash
set -euo pipefail

REPO="${LACLAUGPT_PUBLIC_REPO:-/scratch/project_2009497/LaclauGPT-Multimodal-Analysis}"
COUNTRY="${1:-finland}"
LIMIT="${2:-25}"

cd "$REPO"
# Step 6 fixes are already merged into main. Pull separately before launching.
if [[ ! -f "roihu_populism.py" || ! -f "scripts/roihu/submit_step_6_discourse_analysis.sh" ]]; then
  echo "ERROR: expected LaclauGPT-Multimodal-Analysis repository not found: $REPO" >&2
  exit 2
fi
if ! grep -q 'LACLAUGPT_STEP6_MAX_ATTEMPTS' roihu_populism.py; then
  echo "ERROR: Step 6 retry fix not found. Run 'git pull origin main' first." >&2
  exit 2
fi

export LACLAUGPT_STEP6_NUM_PREDICT="${LACLAUGPT_STEP6_NUM_PREDICT:-8192}"
export LACLAUGPT_STEP6_MAX_ATTEMPTS="${LACLAUGPT_STEP6_MAX_ATTEMPTS:-3}"

# Runs existing cumulative-country checkpoint contract.
# Step 6 environment must already be installed.
bash scripts/roihu/submit_step_6_discourse_analysis.sh --country "$COUNTRY" --limit "$LIMIT"
