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
# All EP24 step logs share one directory. The latest canonical .log is a symlink
# to a job-specific immutable log so concurrent submissions never truncate history.
LOG_DIR="${EP24_LOG_DIR:-${LACLAUGPT_EP24_LOG_DIR:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/logs}}"
case "${LOG_DIR}" in /*) ;; *) echo "EP24_LOG_DIR must be absolute" >&2; exit 2;; esac
COUNTRY=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --country) ((i+1 < ${#ARGS[@]})) || { echo "--country requires a value" >&2; exit 2; }
               COUNTRY="${ARGS[i+1]}"; i=$((i+1)) ;;
    --country=*) COUNTRY="${ARGS[i]#--country=}" ;;
  esac
done
COUNTRY="${COUNTRY,,}"
case "${COUNTRY}" in
  fi|finland) COUNTRY=finland ;;
  pl|poland) COUNTRY=poland ;;
  pt|portugal) COUNTRY=portugal ;;
  *) echo "EP24 log naming requires --country finland|poland|portugal" >&2; exit 2 ;;
esac
STEP_NAME="$(roihu_step_name "${STEP}")"
LOG_STEM="${STEP_NAME}-${COUNTRY}"
mkdir -p "${LOG_DIR}"
chmod 700 "${LOG_DIR}"
export LACLAUGPT_EP24_LOG_DIR="${LOG_DIR}"
export EP24_LOG_DIR="${LOG_DIR}"

SBATCH_ARGS=(--parsable --output="${LOG_DIR}/${LOG_STEM}-%j.log" --error="${LOG_DIR}/${LOG_STEM}-%j.log" --open-mode=append --job-name="ep24-step-${STEP}" --account="${CSC_PROJECT}" --partition="${PARTITION}" --cpus-per-task="${CPUS}" --time="${TIME}")
[[ "${MEM}" != "0" && -n "${MEM}" ]] && SBATCH_ARGS+=(--mem="${MEM}")
[[ -n "${GRES}" ]] && SBATCH_ARGS+=(--gres="${GRES}")
[[ -n "${LACLAUGPT_SBATCH_DEPENDENCY:-}" ]] && SBATCH_ARGS+=(--dependency="${LACLAUGPT_SBATCH_DEPENDENCY}")

echo "Submitting EP24 Step ${STEP} ($(roihu_step_name "${STEP}")) partition=${PARTITION} cpus=${CPUS} gres=${GRES:-none} time=${TIME}" >&2
job_id="$(sbatch "${SBATCH_ARGS[@]}" --export=ALL "${SCRIPT_DIR}/step_runner.sbatch" "$@")"
job_id="${job_id%%;*}"
# Canonical link points to the newest submission; previous and concurrent job logs
# stay separate and readable by their job IDs.
ln -sfn "${LOG_STEM}-${job_id}.log" "${LOG_DIR}/.${LOG_STEM}.latest.$"
mv -f "${LOG_DIR}/.${LOG_STEM}.latest.$" "${LOG_DIR}/${LOG_STEM}.log"
echo "EP24 log: ${LOG_DIR}/${LOG_STEM}.log (job ${job_id})" >&2
echo "Job archive: ${LOG_DIR}/${LOG_STEM}-${job_id}.log" >&2
echo "Follow live: tail -f ${LOG_DIR}/${LOG_STEM}-${job_id}.log" >&2
echo "${job_id}"
