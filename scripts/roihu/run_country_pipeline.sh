#!/usr/bin/env bash
# Submit an ordered, restartable Step 1-9 SLURM dependency chain for one country.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/roihu/run_country_pipeline.sh COUNTRY [--limit N] [--retry-errors] [--from-step N] [--to-step N] [--include-paused]
Example: bash scripts/roihu/run_country_pipeline.sh portugal --limit 0 --retry-errors
Each stage starts only after the previous stage's SLURM job exits successfully.
Completed MongoDB records are preserved; this chain never forces a reset.

Scope: steps 1-7 by default (#346). Steps 8 (SNA) and 9 (RDF) are PAUSED for the active
EP24 FI/PL/PT run: their code, tests and stored outputs remain, but no job is submitted for
them unless you ask explicitly with --to-step 8/9 --include-paused.
EOF
}
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
case "${1}" in -h|--help) usage; exit 0 ;; esac
COUNTRY="${1}"
shift
[[ "${COUNTRY}" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || {
  echo "Invalid country: ${COUNTRY}" >&2; exit 2;
}
LIMIT=0
FROM_STEP=1
# #346: 7 is the terminal stage of a normal run. Steps 8-9 are paused, not deleted.
TO_STEP=7
PAUSED_LAST_STEP=9
INCLUDE_PAUSED=0
RETRY=0
while (($#)); do
  case "${1}" in
    --limit)
      [[ $# -ge 2 && "${2}" =~ ^[0-9]+$ ]] || { echo "--limit requires a non-negative integer" >&2; exit 2; }
      LIMIT="${2}"; shift 2 ;;
    --from-step)
      [[ $# -ge 2 && "${2}" =~ ^[1-9]$ ]] || { echo "--from-step must be 1..9" >&2; exit 2; }
      FROM_STEP="${2}"; shift 2 ;;
    --to-step)
      [[ $# -ge 2 && "${2}" =~ ^[1-9]$ ]] || { echo "--to-step must be 1..9" >&2; exit 2; }
      TO_STEP="${2}"; shift 2 ;;
    --include-paused) INCLUDE_PAUSED=1; shift ;;
    --retry-errors) RETRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
  esac
done
(( FROM_STEP <= TO_STEP )) || { echo "--from-step must be <= --to-step" >&2; exit 2; }
if (( TO_STEP > 7 )) && (( INCLUDE_PAUSED == 0 )); then
  echo "Steps 8-9 are paused for the active EP24 run (#346): the default scope ends at Step 7." >&2
  echo "Their code and stored outputs are intact. To run them deliberately:" >&2
  echo "  bash $0 ${COUNTRY} --from-step 8 --to-step 9 --include-paused" >&2
  exit 2
fi
if (( TO_STEP > PAUSED_LAST_STEP )); then
  echo "--to-step must be <= ${PAUSED_LAST_STEP}" >&2; exit 2
fi
command -v sbatch >/dev/null 2>&1 || { echo "sbatch unavailable; run from Roihu login node" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${LACLAUGPT_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private}/analysis/ep24_reprocess}"
LOG_ROOT="${LACLAUGPT_EP24_LOG_ROOT:-${PRIVATE_ROOT}/logs}"
mkdir -p "${LOG_ROOT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${COUNTRY}_${FROM_STEP}-${TO_STEP}_$$"
MANIFEST="${LOG_ROOT}/pipeline_${RUN_ID}.tsv"
printf 'country\tstep\tjob_id\tdependency\n' > "${MANIFEST}"
# The scope travels with the run: the exporter and the audit read this to tell "paused on purpose"
# apart from "missing or failed", without rewriting any historical coverage (#346).
export LACLAUGPT_PAUSED_STAGES="8,9"
if (( TO_STEP < 8 )); then
  printf 'country=%s\nsteps=%s-%s\nlast_step=%s\npaused_steps=8,9\nreason=%s\n' \
    "${COUNTRY}" "${FROM_STEP}" "${TO_STEP}" "${TO_STEP}" \
    "Steps 8-9 paused for the active EP24 run (#346)" > "${LOG_ROOT}/pipeline_${RUN_ID}.scope"
  echo "scope: steps ${FROM_STEP}-${TO_STEP} for ${COUNTRY}; Steps 8-9 paused (#346)" >&2
else
  echo "scope: steps ${FROM_STEP}-${TO_STEP} for ${COUNTRY}, INCLUDING paused Steps 8-9 (explicit opt-in)" >&2
fi
PREVIOUS=""
RETRY_ARGS=()
(( RETRY )) && RETRY_ARGS+=(--retry-errors)

for (( STEP=FROM_STEP; STEP<=TO_STEP; STEP++ )); do
  # Submit through the maintained wrapper, with SLURM flags applied before
  # the batch script rather than accidentally forwarded to the Python CLI.
  DEPENDENCY=""
  [[ -n "${PREVIOUS}" ]] && DEPENDENCY="afterok:${PREVIOUS}"
  OUTPUT="$(LACLAUGPT_SBATCH_DEPENDENCY="${DEPENDENCY}" \
    LACLAUGPT_SBATCH_PARSABLE=1 \
    bash "${SCRIPT_DIR}/run_step.sh" "${STEP}" \
    --country "${COUNTRY}" --limit "${LIMIT}" "${RETRY_ARGS[@]}")"
  JOB_ID="$(printf '%s\n' "${OUTPUT}" | tail -n 1)"
  JOB_ID="${JOB_ID%%;*}"
  [[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "Failed to parse Step ${STEP} SLURM ID from: ${OUTPUT}" >&2
    echo "Previously submitted jobs remain in SLURM; inspect ${MANIFEST}" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\n' "${COUNTRY}" "${STEP}" "${JOB_ID}" "${PREVIOUS:-none}" >> "${MANIFEST}"
  echo "Step ${STEP}: job ${JOB_ID}${PREVIOUS:+ (afterok:${PREVIOUS})}"
  PREVIOUS="${JOB_ID}"
done

echo "All requested stages submitted for ${COUNTRY}."
echo "Manifest: ${MANIFEST}"
echo "Logs: ${LOG_ROOT}/ep24_s<step>_<jobid>.{out,err}"
echo "SLURM: squeue -u \"${USER}\""
echo "Failed stages block their dependents; rerun this command after fixing the failure."
