#!/usr/bin/env bash
# Shared launcher for the numbered EP24 Roihu pipeline steps (issue #64).
#
# Source this from a thin `run_step_N.sh` wrapper. It resolves the public/private
# roots, loads the CSC modules, activates the venv, validates the configuration,
# creates the directories the job needs, and SUBMITS the matching sbatch job --
# it does not hold the SSH session open, so a dropped connection cannot kill a
# job that has already been queued.
#
# Everything machine-specific comes from the environment, never from a literal in
# this file. No credentials are read, printed or exported here.

set -euo pipefail

LAUNCHER_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd -- "${LAUNCHER_LIB_DIR}/../.." && pwd)}"
LAUNCHER_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-}"
LAUNCHER_VENV="${LACLAUGPT_MULTIMODAL_VENV:-${LAUNCHER_PUBLIC_ROOT}/.venv-roihu-gpu}"
LAUNCHER_SCRATCH="${LACLAUGPT_SCRATCH:-/scratch/project_2009497}"
LAUNCHER_LOGDIR="${LAUNCHER_SCRATCH}/logs"

launcher_log() { printf '[run_step] %s\n' "$*" >&2; }
launcher_die() { printf '[run_step] ERROR: %s\n' "$*" >&2; exit 2; }

# Load private settings if present. The file is expected to export only
# non-secret configuration; secrets stay in the operator's environment.
launcher_load_settings() {
  local settings="${LACLAUGPT_EP24_SETTINGS:-${LAUNCHER_PRIVATE_ROOT}/ep24_reprocess.env}"
  if [[ -n "${LACLAUGPT_EP24_SETTINGS:-}" && ! -f "${settings}" ]]; then
    launcher_die "LACLAUGPT_EP24_SETTINGS points at a missing file: ${settings}"
  fi
  if [[ -f "${settings}" ]]; then
    launcher_log "loading settings from ${settings}"
    # shellcheck disable=SC1090
    set -a; source "${settings}"; set +a
  else
    launcher_log "no settings file at ${settings} (using the current environment)"
  fi
}

launcher_require() {
  local missing=0
  [[ -n "${LAUNCHER_PRIVATE_ROOT}" ]] || { launcher_log "LACLAUGPT_MULTIMODAL_PRIVATE_ROOT is not set"; missing=1; }
  [[ -d "${LAUNCHER_PRIVATE_ROOT}" ]] || { launcher_log "private root does not exist: ${LAUNCHER_PRIVATE_ROOT}"; missing=1; }
  [[ -x "${LAUNCHER_VENV}/bin/python" ]] || { launcher_log "venv python not executable: ${LAUNCHER_VENV}/bin/python"; missing=1; }
  [[ -n "${LACLAUGPT_MONGO_URI:-}" ]] || launcher_log "LACLAUGPT_MONGO_URI is not set; MongoDB-backed stages will fail"
  (( missing == 0 )) || launcher_die "required configuration is missing (see above)"
}

launcher_prepare_dirs() {
  mkdir -p "${LAUNCHER_LOGDIR}" \
           "${LAUNCHER_SCRATCH}/hf-cache" \
           "${LAUNCHER_PRIVATE_ROOT}"/{outputs,csv,database,logs}
}

launcher_modules() {
  if command -v module >/dev/null 2>&1; then
    module --force purge
    module load python-pytorch
    module load ffmpeg
    launcher_log "loaded CSC modules: python-pytorch, ffmpeg"
  else
    launcher_log "no 'module' command (not on CSC); skipping module load"
  fi
  unset PYTHONPATH PYTHONHOME || true
}

launcher_banner() {
  local step="$1"
  cat >&2 <<BANNER
[run_step] ── EP24 Roihu step ${step} ─────────────────────────────
[run_step] public root   : ${LAUNCHER_PUBLIC_ROOT}
[run_step] private root  : ${LAUNCHER_PRIVATE_ROOT}
[run_step] venv          : ${LAUNCHER_VENV}
[run_step] logs          : ${LAUNCHER_LOGDIR}
[run_step] countries     : ${LACLAUGPT_COUNTRIES:-<all, priority order>}
[run_step] max rows      : ${LACLAUGPT_MAX_ROWS:-100}
[run_step] ────────────────────────────────────────────────────────
BANNER
}

# Submit the matching sbatch job and print where it went. `sbatch` returns
# immediately, so this launcher never holds the connection open.
launcher_submit() {
  local step="$1"
  local sbatch_file="${LAUNCHER_PUBLIC_ROOT}/scripts/roihu/step_${step}_roihu_${STEP_NAME}.sbatch"
  [[ -f "${sbatch_file}" ]] || launcher_die "missing sbatch file: ${sbatch_file}"

  launcher_log "submitting ${sbatch_file}"
  local job_id
  job_id="$(sbatch --parsable "${sbatch_file}")" || launcher_die "sbatch submission failed"

  cat >&2 <<DONE
[run_step] submitted job ${job_id}
[run_step] stdout: ${LAUNCHER_LOGDIR}/ep24_s${step}_${job_id}.out
[run_step] stderr: ${LAUNCHER_LOGDIR}/ep24_s${step}_${job_id}.err
[run_step] watch : squeue -j ${job_id}
[run_step] tail  : tail -f ${LAUNCHER_LOGDIR}/ep24_s${step}_${job_id}.err
DONE
}

# Entry point used by every run_step_N.sh wrapper.
launch_step() {
  local step="$1" name="$2"
  STEP_NAME="${name}"
  export LACLAUGPT_EP24_STEP="${step}"
  launcher_load_settings
  launcher_require
  launcher_modules
  launcher_prepare_dirs
  launcher_banner "${step}"
  launcher_submit "${step}"
}
