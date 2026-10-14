#!/usr/bin/env bash
set -euo pipefail
# Submit one country as a strict Slurm afterok chain: 1 -> 2 -> ... -> 9.
# All job submissions occur now; execution never overlaps between stages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { echo "Usage: bash scripts/roihu/run_country_sequential.sh COUNTRY [--limit N] [--retry-errors]" >&2; }
[[ $# -ge 1 ]] || { usage; exit 2; }
COUNTRY="$1"; shift
[[ "$COUNTRY" =~ ^[a-zA-Z][a-zA-Z_-]*$ ]] || { echo "Invalid country: $COUNTRY" >&2; exit 2; }
LIMIT=0
RETRY=0
while (($#)); do
  case "$1" in
    --limit) [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { usage; exit 2; }; LIMIT="$2"; shift 2 ;;
    --retry-errors) RETRY=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
command -v sbatch >/dev/null || { echo "sbatch not found; run on Roihu" >&2; exit 2; }
source "${SCRIPT_DIR}/roihu_step_profiles.sh"
NAMES=( "" preprocess frame video summary postprocess discourse_analysis discourse_network_analysis social_network_analysis rdf )
# Validate all environments before any submission, preventing partial chains.
for step in {1..9}; do
  venv="$(roihu_step_venv "$step")"
  [[ -x "$venv/bin/python" ]] || { echo "Missing step $step venv: $venv" >&2; exit 2; }
done
previous=""
for step in {1..9}; do
  if [[ -n "$previous" ]]; then
    export LACLAUGPT_SBATCH_DEPENDENCY="afterok:${previous}"
  else
    unset LACLAUGPT_SBATCH_DEPENDENCY || true
  fi
  submitter="${SCRIPT_DIR}/submit_step_${step}_${NAMES[$step]}.sh"
  args=(--country "$COUNTRY" --limit "$LIMIT")
  if (( RETRY )); then args+=(--retry-errors); fi
  job_id="$(bash "$submitter" "${args[@]}")"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $job_id" >&2; exit 1; }
  echo "step=$step country=$COUNTRY job=$job_id ${previous:+afterok:$previous}" >&2
  previous="$job_id"
done
unset LACLAUGPT_SBATCH_DEPENDENCY || true
echo "Submitted serial steps 1-9 for $COUNTRY (final job $previous). Each step starts only if the previous step exits successfully."
