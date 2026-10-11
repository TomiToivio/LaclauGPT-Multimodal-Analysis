#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
INPUT="$1"
N="$(test -n "${2:-}" && echo "$2" || echo 10)"
test -f "$INPUT" || { echo "Missing Finnish Step 4 CSV: $INPUT" >&2; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || exit 2
INPUT="$(realpath "$INPUT")"
RUN_ROOT=".local/step5-fi-test-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(realpath "$RUN_ROOT")"
export INPUT N RUN_ROOT
python3 - <<'PY'
import csv, os
from pathlib import Path
src=Path(os.environ["INPUT"])
dst=Path(os.environ["RUN_ROOT"])/"step4_fi_sample.csv"
with src.open(newline="",encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    fields=reader.fieldnames
    if not fields or "summary_analysis" not in fields:
        raise SystemExit("Step 4 summary_analysis column missing")
    rows=[]
    for r in reader:
        if (r.get("country") or "").strip().lower() not in ("", "fi", "finland", "suomi"):
            continue
        if not (r.get("summary_analysis") or "").strip():
            continue
        r=dict(r)
        # Step 5 DELETE routing unlinks these local paths. Remove from TEST COPY only.
        for field in ("frame_file","vllm_video_local_path","vllm_video_analysis_path"):
            if field in r: r[field]=""
        rows.append(r)
        if len(rows)>=int(os.environ["N"]): break
if not rows: raise SystemExit("No Finnish records with a Step 4 summary")
with dst.open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Prepared {len(rows)} Finnish records in {dst}")
PY
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_LANGUAGES=fi
export LACLAUGPT_INPUT_CSV="$RUN_ROOT/step4_fi_sample.csv"
export LACLAUGPT_OUTPUT_CSV="$RUN_ROOT/step5_fi_output.csv"
export LACLAUGPT_MEMORY_DB="$RUN_ROOT/ep24_memory.sqlite3"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="$RUN_ROOT"
export LACLAUGPT_MONGO_ENABLED=0
export LACLAUGPT_MAX_ROWS="$N"
unset LACLAUGPT_REDIS_URL
echo "Running Finnish Step 5 with model: ${LACLAUGPT_MULTIMODAL_MODEL:-repository-default}"
python3 step_5_roihu_postprocess.py --country finland --limit "$N"
echo "Review CSV and routing audit files in $RUN_ROOT"
echo "Inspect logs/postprocess.log for extraction and normalization errors"
