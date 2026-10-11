#!/usr/bin/env bash
# Isolated Finnish-only Step 4 quality gate for issue #236.
# Usage: bash scripts/ep24/test_step4_fi.sh /absolute/path/to/step_03_video.csv [10]
set -euo pipefail
cd "$(dirname "$0")/../.."
INPUT="${1:?Provide the existing Finnish Step 3 cumulative CSV path}"
N="${2:-10}"
[[ -f "$INPUT" ]] || { echo "Missing input CSV: $INPUT" >&2; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "Sample size must be positive" >&2; exit 2; }
INPUT="$(realpath "$INPUT")"
RUN_ROOT="${LACLAUGPT_STEP4_FI_TEST_ROOT:-$PWD/.local/step4-fi-test-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(realpath "$RUN_ROOT")"
[[ "$RUN_ROOT/" != "$(dirname "$INPUT")/" ]] || { echo "Test root cannot be input directory" >&2; exit 2; }
export INPUT N RUN_ROOT
python3 - <<'PY'
import csv, os
from pathlib import Path
src=Path(os.environ["INPUT"])
dst=Path(os.environ["RUN_ROOT"])/"step3_fi_sample.csv"
with src.open(newline="", encoding="utf-8-sig") as h:
    reader=csv.DictReader(h)
    fields=reader.fieldnames
    if not fields: raise SystemExit("Input CSV has no header")
    rows=[]
    for row in reader:
        country=(row.get("country") or "").strip().casefold()
        if country and country not in ("fi", "finland", "suomi"):
            continue
        rows.append(row)
        if len(rows)>=int(os.environ["N"]): break
if not rows: raise SystemExit("No eligible Finnish records found")
with dst.open("w", newline="", encoding="utf-8") as h:
    writer=csv.DictWriter(h, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Sample: {len(rows)} videos -> {dst}")
PY
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_LANGUAGES=fi
export LACLAUGPT_INPUT_CSV="$RUN_ROOT/step3_fi_sample.csv"
export LACLAUGPT_OUTPUT_CSV="$RUN_ROOT/step4_fi_output.csv"
export LACLAUGPT_SUMMARY_SQLITE="$RUN_ROOT/summary_cache.sqlite3"
export LACLAUGPT_MONGO_ENABLED=0
export LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL=0
export LACLAUGPT_MAX_ROWS="$N"
unset LACLAUGPT_REDIS_URL
export LACLAUGPT_SUMMARY_EVIDENCE_CHARS="${LACLAUGPT_SUMMARY_EVIDENCE_CHARS:-20000}"
export LACLAUGPT_SUMMARY_NUM_CTX="${LACLAUGPT_SUMMARY_NUM_CTX:-32768}"
export LACLAUGPT_SUMMARY_NUM_PREDICT="${LACLAUGPT_SUMMARY_NUM_PREDICT:-2048}"
echo "Model: ${LACLAUGPT_MULTIMODAL_MODEL:-repository default}; input=$LACLAUGPT_INPUT_CSV output=$LACLAUGPT_OUTPUT_CSV"
python3 step_4_roihu_summary.py --country finland --limit "$N"
echo "Output: $LACLAUGPT_OUTPUT_CSV"
echo "Log: logs/summary.log (shared log); compare summary_analysis with the preserved baseline using video_id."
