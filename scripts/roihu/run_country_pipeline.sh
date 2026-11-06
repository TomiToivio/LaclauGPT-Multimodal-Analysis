#!/usr/bin/env bash
# Submit an ordered, restartable Step 1-9 SLURM dependency chain for one country.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/roihu/run_country_pipeline.sh COUNTRY [--limit N] [--retry-errors]
       [--from-step N] [--to-step N] [--require-dataset NAME]
Example: bash scripts/roihu/run_country_pipeline.sh portugal --limit 0 --retry-errors
Each stage starts after the previous stage terminates; failures remain recorded for selective retry.
Completed MongoDB records are preserved; this chain never forces a reset.

--require-dataset NAME  refuse to submit unless the environment resolves to that dataset.
                        Use it for a pilot resume: without it, an env file that omits
                        LACLAUGPT_DATASET silently resolves to ep2024_reprocess (full corpus).
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
TO_STEP=7   # #346: Steps 8-9 are paused; opt in with EP24_INCLUDE_PAUSED_STEPS=1
RETRY=0
REQUIRE_DATASET=""
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
    --retry-errors) RETRY=1; shift ;;
    --require-dataset)
      [[ $# -ge 2 && -n "${2}" ]] || { echo "--require-dataset requires a name" >&2; exit 2; }
      REQUIRE_DATASET="${2}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
  esac
done
(( FROM_STEP <= TO_STEP )) || { echo "--from-step must be <= --to-step" >&2; exit 2; }

# #346: Steps 8-9 are paused. Submitting them requires an explicit, visible opt-in - never a slip of
# --to-step.
if (( TO_STEP >= 8 )); then
  case "${EP24_INCLUDE_PAUSED_STEPS:-0}" in
    1|true|yes|on) echo "NOTE: paused steps 8-9 explicitly enabled for this run" >&2 ;;
    *) echo "Step range ends at ${TO_STEP}, but Steps 8-9 are paused (#346)." >&2
       echo "Use --to-step 7 for a normal run, or set EP24_INCLUDE_PAUSED_STEPS=1 deliberately." >&2
       exit 2 ;;
  esac
fi
command -v sbatch >/dev/null 2>&1 || { echo "sbatch unavailable; run from Roihu login node" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${LACLAUGPT_PRIVATE_ROOT:-/scratch/project_2009497/LaclauGPT-Private}/analysis/ep24_reprocess}"
LOG_ROOT="${LACLAUGPT_EP24_LOG_ROOT:-${PRIVATE_ROOT}/logs}"
mkdir -p "${LOG_ROOT}"
# #379: resolve the dataset the way the Slurm job will resolve it, and refuse to submit a guess.
ENV_FILE="${LACLAUGPT_EP24_ENV_FILE:-${PRIVATE_ROOT}/.env}"
RESOLVED_DATASET="$(
  set -a
  if [[ -f "${ENV_FILE}" ]]; then . "${ENV_FILE}" >/dev/null 2>&1 || exit 3; fi
  set +a
  printf '%s' "${LACLAUGPT_DATASET:-ep2024_reprocess}"
)" || { echo "Cannot read ${ENV_FILE}" >&2; exit 2; }
echo "Dataset: ${RESOLVED_DATASET} (env file: ${ENV_FILE})" >&2
if [[ -n "${REQUIRE_DATASET}" && "${RESOLVED_DATASET}" != "${REQUIRE_DATASET}" ]]; then
  echo "Refusing to submit: this environment resolves to dataset '${RESOLVED_DATASET}'," >&2
  echo "but --require-dataset ${REQUIRE_DATASET} was requested. Export LACLAUGPT_DATASET in" >&2
  echo "${ENV_FILE} (not just in your shell) so every Slurm job inherits it." >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${COUNTRY}_${FROM_STEP}-${TO_STEP}_$$"
MANIFEST="${LOG_ROOT}/pipeline_${RUN_ID}.tsv"
printf '# dataset: %s\n' "${RESOLVED_DATASET}" > "${MANIFEST}"
printf 'country\tstep\tjob_id\tdependency\n' >> "${MANIFEST}"
PREVIOUS=""
RETRY_ARGS=()
(( RETRY )) && RETRY_ARGS+=(--retry-errors)

for (( STEP=FROM_STEP; STEP<=TO_STEP; STEP++ )); do
  # Submit through the maintained wrapper, with SLURM flags applied before
  # the batch script rather than accidentally forwarded to the Python CLI.
  DEPENDENCY=""
  [[ -n "${PREVIOUS}" ]] && DEPENDENCY="afterany:${PREVIOUS}"
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
  echo "Step ${STEP}: job ${JOB_ID}${PREVIOUS:+ (afterany:${PREVIOUS})}"
  PREVIOUS="${JOB_ID}"
done

echo "All requested stages submitted for ${COUNTRY}."
echo "Manifest: ${MANIFEST}"
echo "Logs: ${LOG_ROOT}/ep24_s<step>_<jobid>.{out,err}"
echo "SLURM: squeue -u \"${USER}\""
echo "Failed rows remain auditable and retryable; inspect the stage logs before selective reruns."
