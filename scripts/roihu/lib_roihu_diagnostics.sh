#!/usr/bin/env bash
# Shared startup diagnostics for the Roihu step jobs (issue #188).
#
# Source this from a step sbatch script after `set -euo pipefail` and after the
# environment is configured, so a failed run can be reconstructed from the log
# alone: which host, which job, which GPU, which code revision, which runtime.
#
#   source "${PUBLIC_ROOT}/scripts/roihu/lib_roihu_diagnostics.sh"
#   roihu_diagnostics <step-number>
#
# Prints only non-secret information. Private settings are reported by NAME and
# never by value (see ep24_settings.py).

roihu_diagnostics() {
  local step="${1:-?}"
  echo "=== EP24 step ${step} start ==="
  echo "hostname        : $(hostname)"
  echo "date            : $(date -Is)"
  echo "slurm_job_id    : ${SLURM_JOB_ID:-<not under slurm>}"
  echo "slurm_partition : ${SLURM_JOB_PARTITION:-<none>}"
  echo "slurm_cpus      : ${SLURM_CPUS_PER_TASK:-<none>}"
  echo "public_root     : ${PUBLIC_ROOT:-<unset>}"
  echo "private_root    : ${PRIVATE_ROOT:-<unset>}"
  if git -C "${PUBLIC_ROOT:-.}" rev-parse --short HEAD >/dev/null 2>&1; then
    echo "git_commit      : $(git -C "${PUBLIC_ROOT}" rev-parse --short HEAD) ($(git -C "${PUBLIC_ROOT}" rev-parse --abbrev-ref HEAD))"
    if [[ -n "$(git -C "${PUBLIC_ROOT}" status --porcelain 2>/dev/null)" ]]; then
      echo "git_dirty       : yes (results may not be reproducible from the commit)"
    fi
  else
    echo "git_commit      : <not a git checkout>"
  fi
  echo "python          : $(python3 --version 2>&1)"
  echo "python_path     : $(command -v python3 || echo '<none>')"
  echo "hf_home         : ${HF_HOME:-<unset>}"
  echo "tmpdir          : ${TMPDIR:-<unset>}"
  echo "ffmpeg          : $(ffmpeg -version 2>/dev/null | head -1 || echo '<not found>')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu             : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo '<nvidia-smi failed>')"
    echo "cuda            : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
  else
    echo "gpu             : nvidia-smi not present"
  fi
  for pkg in pandas torch numpy cv2 pymongo redis; do
    python3 - "$pkg" <<'PY' 2>/dev/null || echo "package ${pkg:0} : <not importable>"
import importlib, sys
name = sys.argv[1]
mod = importlib.import_module(name)
print(f"package {name:<8}: {getattr(mod, '__version__', 'unknown')}")
PY
  done
  echo "=== EP24 step ${step} diagnostics end ==="
}

roihu_completion_summary() {
  local step="${1:-?}" rc="${2:-$?}"
  echo "=== EP24 step ${step} finished rc=${rc} at $(date -Is) ==="
  if [[ "${rc}" -ne 0 ]]; then
    echo "STEP FAILED: inspect ${SLURM_JOB_ID:+job ${SLURM_JOB_ID} }stderr for the first traceback."
  fi
  return "${rc}"
}
