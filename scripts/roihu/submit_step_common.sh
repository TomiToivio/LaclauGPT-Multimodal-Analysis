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
mkdir -p "/scratch/${CSC_PROJECT}/logs"

SBATCH_ARGS=(--parsable --account="${CSC_PROJECT}" --partition="${PARTITION}" --cpus-per-task="${CPUS}" --time="${TIME}")
[[ "${MEM}" != "0" && -n "${MEM}" ]] && SBATCH_ARGS+=(--mem="${MEM}")
[[ -n "${GRES}" ]] && SBATCH_ARGS+=(--gres="${GRES}")
[[ -n "${LACLAUGPT_SBATCH_DEPENDENCY:-}" ]] && SBATCH_ARGS+=(--dependency="${LACLAUGPT_SBATCH_DEPENDENCY}")

echo "Submitting EP24 Step ${STEP} ($(roihu_step_name "${STEP}")) partition=${PARTITION} cpus=${CPUS} gres=${GRES:-none} time=${TIME}" >&2
job_id="$(sbatch "${SBATCH_ARGS[@]}" --export=ALL "${SCRIPT_DIR}/step_runner.sbatch" "$@")"
job_id="${job_id%%;*}"
echo "${job_id}"
