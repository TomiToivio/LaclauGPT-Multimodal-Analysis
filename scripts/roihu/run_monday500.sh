#!/usr/bin/env bash
set -euo pipefail
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
source scripts/roihu/roihu_job_common.sh
[[ "${LACLAUGPT_MONGO_ENABLED:-0}" == 1 ]] || { echo 'Mongo required' >&2; exit 2; }
RUN_DIR="${RUN_DIR:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/monday500_20261031}"
DATASET="${DATASET:-ep24_monday500_20261031}"
PYTHON_SCRIPT=scripts/ep24/ep24_monday500.py
mkdir -p "$RUN_DIR"
if [[ "${APPROVE_IMPORT:-}" != YES ]]; then
  python3 "$PYTHON_SCRIPT" prepare --run-dir "$RUN_DIR" --dataset "$DATASET"
  echo "Inspect $RUN_DIR/manifest.json then rerun with APPROVE_IMPORT=YES"
  exit 0
fi
python3 "$PYTHON_SCRIPT" prepare --run-dir "$RUN_DIR" --dataset "$DATASET" --apply
export LACLAUGPT_DATASET="$DATASET"
export LACLAUGPT_EP24_ENV_FILE=/dev/null
export LACLAUGPT_EP24_NONBLOCKING=0
export LACLAUGPT_STAGE_BATCH_SIZE=25
export LACLAUGPT_EP24_OUTPUT_ROOT="$RUN_DIR/outputs"
mkdir -p "$RUN_DIR/outputs"
printf 'country\tstep\tjob_id\n' > "$RUN_DIR/jobs.tsv"
for country in finland poland portugal; do
  export LACLAUGPT_COUNTRY="$country"
  export LACLAUGPT_SAMPLE_IDS_FILE="$RUN_DIR/${country}_ids.txt"
  [[ "$(wc -l < "$LACLAUGPT_SAMPLE_IDS_FILE")" -eq 500 ]] || { echo 'Expected 500 selected IDs' >&2; exit 2; }
  previous=""
  for step in 1 2 3 4 5 6 7; do
    case "$step" in
      1) suffix=preprocess;; 2) suffix=frame;; 3) suffix=video;;
      4) suffix=summary;; 5) suffix=postprocess;;
      6) suffix=discourse_analysis;; 7) suffix=discourse_network_analysis;;
    esac
    if [[ -n "$previous" ]]; then
      export LACLAUGPT_SBATCH_DEPENDENCY="afterok:$previous"
    else
      unset LACLAUGPT_SBATCH_DEPENDENCY || true
    fi
    job="$(bash "scripts/roihu/submit_step_${step}_${suffix}.sh" --country "$country" --limit 500 --retry-errors)"
    printf '%s\t%d\t%s\n' "$country" "$step" "$job" | tee -a "$RUN_DIR/jobs.tsv"
    previous="$job"
  done
done
unset LACLAUGPT_SBATCH_DEPENDENCY || true
echo "After jobs complete: python3 $PYTHON_SCRIPT export --run-dir '$RUN_DIR' --count 500"
