#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${PUBLIC_ROOT}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${CSC_PROJECT}/LaclauGPT-Private/analysis/ep24_reprocess}"
PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
INPUT_ROOT="${LACLAUGPT_EP24_INPUT_ROOT:-${PRIVATE_ROOT}/data/to_reprocess}"
SEED=209
MODE=""

usage() {
  echo "usage: run_pipeline.sh (--test|--full) [--seed N] [--input-root PATH] [--run-root PATH]" >&2
}

RUN_ROOT="${LACLAUGPT_PIPELINE_RUN_ROOT:-${PRIVATE_ROOT}/runs}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test|--full)
      [[ -z "${MODE}" ]] || { echo "--test and --full are mutually exclusive" >&2; exit 2; }
      MODE="${1#--}"
      shift ;;
    --seed) SEED="${2:?--seed needs a value}"; shift 2 ;;
    --input-root) INPUT_ROOT="${2:?--input-root needs a path}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?--run-root needs a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "${MODE}" ]] || { usage; exit 2; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found; run on CSC Roihu." >&2; exit 2; }
[[ -d "${INPUT_ROOT}" ]] || { echo "input root not found: ${INPUT_ROOT}" >&2; exit 2; }

GIT_SHA="$(git -C "${PUBLIC_ROOT}" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="ep24_${MODE}_${STAMP}_${GIT_SHA:0:8}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

PREP_ARGS=(--input-root "${INPUT_ROOT}" --run-dir "${RUN_DIR}" --seed "${SEED}")
if [[ "${MODE}" == "test" ]]; then PREP_ARGS+=(--test); else PREP_ARGS+=(--full); fi
EFFECTIVE_INPUT_ROOT="$(python3 "${SCRIPT_DIR}/prepare_pipeline_run.py" "${PREP_ARGS[@]}")"
mapfile -t COUNTRIES < "${RUN_DIR}/countries.txt"
[[ ${#COUNTRIES[@]} -gt 0 ]] || { echo "no countries selected" >&2; exit 2; }

export LACLAUGPT_EP24_INPUT_ROOT="${EFFECTIVE_INPUT_ROOT}"
export LACLAUGPT_EP24_OUTPUT_ROOT="${LACLAUGPT_EP24_OUTPUT_ROOT:-${RUN_DIR}/outputs}"
export LACLAUGPT_PIPELINE_RUN_ID="${RUN_ID}"
mkdir -p "${LACLAUGPT_EP24_OUTPUT_ROOT}"

# Exact test membership is a CSV-chain property. Avoid an already-populated Mongo
# collection silently widening the test sample beyond the 30-row manifest.
if [[ "${MODE}" == "test" ]]; then
  export LACLAUGPT_MONGO_ENABLED=0
fi

source "${SCRIPT_DIR}/roihu_step_profiles.sh"
for step in {1..9}; do
  venv="$(roihu_step_venv "${step}")"
  [[ -x "${venv}/bin/python" ]] || {
    name="$(roihu_step_name "${step}")"
    echo "Step ${step} environment missing: ${venv}" >&2
    echo "Run: bash ${SCRIPT_DIR}/setup_step_${step}_${name}.sh" >&2
    exit 2
  }
done

echo "EP24 pipeline submission"
echo "  mode=${MODE}"
echo "  run_id=${RUN_ID}"
echo "  git_sha=${GIT_SHA}"
echo "  countries=${COUNTRIES[*]}"
echo "  input_root=${EFFECTIVE_INPUT_ROOT}"
echo "  output_root=${LACLAUGPT_EP24_OUTPUT_ROOT}"
if [[ "${MODE}" == "test" ]]; then
  echo "  test_sample=10 random videos each: Finland, Poland, Portugal (30 total)"
  echo "  seed=${SEED}"
else
  echo "  scope=ALL videos in ALL discovered EP24 country CSVs"
fi
echo "  steps=1 2 3 4 5 6 7 8 9"

JOBS_TSV="${RUN_DIR}/jobs.tsv"
printf "country\tstep\tjob_id\n" > "${JOBS_TSV}"
FINAL_IDS=()

submitter_for() {
  case "$1" in
    1) echo "${SCRIPT_DIR}/submit_step_1_preprocess.sh" ;;
    2) echo "${SCRIPT_DIR}/submit_step_2_frame.sh" ;;
    3) echo "${SCRIPT_DIR}/submit_step_3_video.sh" ;;
    4) echo "${SCRIPT_DIR}/submit_step_4_summary.sh" ;;
    5) echo "${SCRIPT_DIR}/submit_step_5_postprocess.sh" ;;
    6) echo "${SCRIPT_DIR}/submit_step_6_discourse_analysis.sh" ;;
    7) echo "${SCRIPT_DIR}/submit_step_7_discourse_network_analysis.sh" ;;
    8) echo "${SCRIPT_DIR}/submit_step_8_social_network_analysis.sh" ;;
    9) echo "${SCRIPT_DIR}/submit_step_9_rdf.sh" ;;
  esac
}

for country in "${COUNTRIES[@]}"; do
  previous=""
  for step in {1..9}; do
    if [[ -n "${previous}" ]]; then
      export LACLAUGPT_SBATCH_DEPENDENCY="afterok:${previous}"
    else
      unset LACLAUGPT_SBATCH_DEPENDENCY || true
    fi
    submitter="$(submitter_for "${step}")"
    job_id="$(bash "${submitter}" --country "${country}" --limit 0)"
    printf "%s\t%s\t%s\n" "${country}" "${step}" "${job_id}" >> "${JOBS_TSV}"
    previous="${job_id}"
  done
  FINAL_IDS+=("${previous}")
done
unset LACLAUGPT_SBATCH_DEPENDENCY || true

FINAL_DEP="afterany:$(IFS=:; echo "${FINAL_IDS[*]}")"
SUMMARY_JOB="$(sbatch --parsable --account="${CSC_PROJECT}" --partition=small --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --dependency="${FINAL_DEP}" --export=ALL,LACLAUGPT_PIPELINE_RUN_DIR="${RUN_DIR}",LACLAUGPT_PIPELINE_JOBS_TSV="${JOBS_TSV}" \
  "${SCRIPT_DIR}/pipeline_summary.sbatch")"
SUMMARY_JOB="${SUMMARY_JOB%%;*}"
echo "${SUMMARY_JOB}" > "${RUN_DIR}/summary_job_id.txt"

python3 - "${RUN_DIR}" "${GIT_SHA}" "${MODE}" "${SUMMARY_JOB}" <<'PY'
import json, pathlib, sys
run_dir=pathlib.Path(sys.argv[1])
manifest=json.loads((run_dir/"manifest.json").read_text())
manifest.update({"git_sha":sys.argv[2],"mode":sys.argv[3],"summary_job_id":sys.argv[4]})
jobs=[]
for line in (run_dir/"jobs.tsv").read_text().splitlines()[1:]:
    country,step,job=line.split("\t")
    jobs.append({"country":country,"step":int(step),"job_id":job})
manifest["jobs"]=jobs
(run_dir/"run.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
PY

echo "Submitted ${#COUNTRIES[@]} country chains, Steps 1-9."
echo "Run metadata: ${RUN_DIR}/run.json"
echo "Final summary job: ${SUMMARY_JOB}"
echo "Final summary will be written to: ${RUN_DIR}/summary.{json,md}"
