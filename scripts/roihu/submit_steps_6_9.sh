#!/usr/bin/env bash
# Submit only EP24 steps 6-9 for independent FI / PL / PT SLURM chains.
# Usage: bash scripts/roihu/submit_steps_6_9.sh [--country fi|pl|pt|all] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNTRY_SELECTION="all"
DRY_RUN=0

usage() {
  cat <<'HELP'
Usage: bash scripts/roihu/submit_steps_6_9.sh [--country fi|pl|pt|all] [--dry-run]

  --country fi|pl|pt|all   Submit one country or all three (default: all).
  --dry-run                Print the four-step chains without calling sbatch.
  --help                   Display this help.

Submits only Steps 6, 7, 8 and 9. Each country's steps run sequentially
(6 -> 7 -> 8 -> 9, afterok dependencies), but country chains are independent
and can run concurrently. Requires Steps 1-5 to be already available in the
configured EP24 upstream data; it does not validate their completion.
HELP
}

while (($#)); do
  case "$1" in
    --country)
      (($# >= 2)) || { echo "--country requires a value" >&2; exit 2; }
      COUNTRY_SELECTION="$2"; shift 2 ;;
    --country=*) COUNTRY_SELECTION="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

COUNTRY_SELECTION="${COUNTRY_SELECTION,,}"
case "$COUNTRY_SELECTION" in
  all) COUNTRIES=(finland poland portugal) ;;
  fi|finland) COUNTRIES=(finland) ;;
  pl|poland) COUNTRIES=(poland) ;;
  pt|portugal) COUNTRIES=(portugal) ;;
  *) echo "Unknown country: $COUNTRY_SELECTION (expected fi, pl, pt or all)" >&2; exit 2 ;;
esac

# The normal per-step submit wrappers own venv/resource validation and log routing.
for step in 6 7 8 9; do
  case "$step" in
    6) SUBMITTERS[6]="$SCRIPT_DIR/submit_step_6_discourse_analysis.sh" ;;
    7) SUBMITTERS[7]="$SCRIPT_DIR/submit_step_7_discourse_network_analysis.sh" ;;
    8) SUBMITTERS[8]="$SCRIPT_DIR/submit_step_8_social_network_analysis.sh" ;;
    9) SUBMITTERS[9]="$SCRIPT_DIR/submit_step_9_rdf.sh" ;;
  esac
  [[ -f "${SUBMITTERS[$step]}" ]] || { echo "Missing submitter for step $step" >&2; exit 2; }
done

if ((DRY_RUN == 0)); then
  command -v sbatch >/dev/null 2>&1 || { echo "sbatch unavailable: submit from a SLURM login node" >&2; exit 2; }
  if [[ -n "${LACLAUGPT_SBATCH_DEPENDENCY:-}" ]]; then
    echo "Refusing inherited LACLAUGPT_SBATCH_DEPENDENCY: unset it to start independent chains" >&2
    exit 2
  fi
fi

echo "EP24: submit Steps 6-9 only for ${COUNTRIES[*]}" >&2
printf 'country\tstep\tjob_id\tdependency\n'
for country in "${COUNTRIES[@]}"; do
  previous=""
  for step in 6 7 8 9; do
    dependency=""
    if [[ -n "$previous" ]]; then
      dependency="afterok:$previous"
    fi
    if ((DRY_RUN)); then
      job_id="DRYRUN-${country}-${step}"
    else
      # Limit 0 means the whole configured corpus, as in run_pipeline.sh.
      # Pass each country's dependency ONLY to its own next step.
      if [[ -n "$dependency" ]]; then
        job_id="$(LACLAUGPT_SBATCH_DEPENDENCY="$dependency" bash "${SUBMITTERS[$step]}" --country "$country" --limit 0)"
      else
        job_id="$(env -u LACLAUGPT_SBATCH_DEPENDENCY bash "${SUBMITTERS[$step]}" --country "$country" --limit 0)"
      fi
      [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid SLURM job id for $country step $step: $job_id" >&2; exit 1; }
    fi
    printf '%s\t%s\t%s\t%s\n' "$country" "$step" "$job_id" "${dependency:--}"
    previous="$job_id"
  done
done
echo "Submission complete; no Steps 1-5 were submitted." >&2
