#!/usr/bin/env bash
set -euo pipefail
STEP="${1:?usage: submit_step_common.sh STEP [step args...]}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${PUBLIC_ROOT}"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${CSC_PROJECT}/LaclauGPT-Private/analysis/ep24_reprocess}"
source "${SCRIPT_DIR}/roihu_step_profiles.sh"

VENV="$(roihu_step_venv "${STEP}")"
[[ -x "${VENV}/bin/python" ]] || {
  echo "Step ${STEP} environment missing: ${VENV}" >&2
  echo "Run: bash ${SCRIPT_DIR}/setup_step_${STEP}_$(roihu_step_name "${STEP}").sh" >&2
  exit 2
}
export LACLAUGPT_STEP="${STEP}"
export LACLAUGPT_STEP_VENV="${VENV}"
if [[ "${STEP}" == "3" ]]; then
  export LACLAUGPT_VLLM_TEST_VENV="${VENV}"
else
  export LACLAUGPT_MULTIMODAL_VENV="${VENV}"
fi

IFS='|' read -r PARTITION CPUS MEM GRES TIME <<<"$(roihu_step_resources "${STEP}")"
PARTITION="${LACLAUGPT_SBATCH_PARTITION:-${PARTITION}}"
CPUS="${LACLAUGPT_SBATCH_CPUS:-${CPUS}}"
MEM="${LACLAUGPT_SBATCH_MEM:-${MEM}}"
GRES="${LACLAUGPT_SBATCH_GRES:-${GRES}}"
TIME="${LACLAUGPT_SBATCH_TIME:-${TIME}}"
# One log directory for Python logs and scheduler logs (#324), configured by EP24_LOG_DIR with the
# legacy name honoured so existing jobs keep working.
LOG_DIR="${EP24_LOG_DIR:-${LACLAUGPT_EP24_LOG_DIR:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/logs}}"
mkdir -p "${LOG_DIR}"
chmod 700 "${LOG_DIR}"
export EP24_LOG_DIR="${LOG_DIR}"

# Canonical step names, identical to ep24_logging.STEP_NAMES; keep the two in step.
case "${STEP}" in
  0) STEP_NAME="import" ;; 1) STEP_NAME="preprocess" ;; 2) STEP_NAME="frame_analysis" ;;
  3) STEP_NAME="video_analysis" ;; 4) STEP_NAME="summary" ;; 5) STEP_NAME="postprocess" ;;
  6) STEP_NAME="laclau_discourse" ;; 7) STEP_NAME="discourse_network" ;;
  8) STEP_NAME="social_network" ;; 9) STEP_NAME="rdf" ;;
  *) echo "unknown step ${STEP}" >&2; exit 2 ;;
esac

# The country is a step argument; the scheduler log name must reflect it or every country's job output
# collides in one file.
COUNTRY="all"
prev=""
for arg in "$@"; do
  case "${prev}" in
    --country|-c) COUNTRY="${arg}" ;;
  esac
  case "${arg}" in
    --country=*) COUNTRY="${arg#--country=}" ;;
  esac
  prev="${arg}"
done
case "${COUNTRY}" in
  fi|fin) COUNTRY="finland" ;; pl|pol) COUNTRY="poland" ;; pt|port) COUNTRY="portugal" ;;
esac
export LACLAUGPT_COUNTRY="${COUNTRY}"

SBATCH_ARGS=(--parsable --output="${LOG_DIR}/${STEP_NAME}-${COUNTRY}-%j.out" --error="${LOG_DIR}/${STEP_NAME}-${COUNTRY}-%j.err" --job-name="ep24-${STEP_NAME}-${COUNTRY}" --account="${CSC_PROJECT}" --partition="${PARTITION}" --cpus-per-task="${CPUS}" --time="${TIME}")
[[ "${MEM}" != "0" && -n "${MEM}" ]] && SBATCH_ARGS+=(--mem="${MEM}")
[[ -n "${GRES}" ]] && SBATCH_ARGS+=(--gres="${GRES}")
[[ -n "${LACLAUGPT_SBATCH_DEPENDENCY:-}" ]] && SBATCH_ARGS+=(--dependency="${LACLAUGPT_SBATCH_DEPENDENCY}")

echo "Submitting EP24 Step ${STEP} ($(roihu_step_name "${STEP}")) partition=${PARTITION} cpus=${CPUS} gres=${GRES:-none} time=${TIME}" >&2
job_id="$(sbatch "${SBATCH_ARGS[@]}" --export=ALL "${SCRIPT_DIR}/step_runner.sbatch" "$@")"
job_id="${job_id%%;*}"
echo "Slurm stdout: ${LOG_DIR}/${STEP_NAME}-${COUNTRY}-${job_id}.out" >&2
echo "Slurm stderr: ${LOG_DIR}/${STEP_NAME}-${COUNTRY}-${job_id}.err" >&2
echo "Follow live: tail -f ${LOG_DIR}/${STEP_NAME}-${COUNTRY}-${job_id}.out ${LOG_DIR}/${STEP_NAME}-${COUNTRY}-${job_id}.err" >&2
echo "${job_id}"
