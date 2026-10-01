#!/usr/bin/env bash
# CSC Roihu bootstrap for the EP24 Qwen3-VL/vLLM video pipeline.
#
# After every reconnect:
#   source /scratch/project_2009497/LaclauGPT-Multimodal-Analysis/scripts/roihu/activate_vllm_video.sh
#
# This file is intentionally sourceable. It configures the *current shell* so
# modules, venv and exported settings survive after the script returns.

_roihu_vllm_fail() {
  printf 'ERROR: %s\n' "$*" >&2
  return 1
}

_roihu_vllm_bootstrap() {
  export CSC_PROJECT="${CSC_PROJECT:-project_2009497}"
  export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${CSC_PROJECT}}"

  export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SCRATCH_ROOT}/LaclauGPT-Multimodal-Analysis}"
  export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${SCRATCH_ROOT}/laclaugpt-multimodal}"
  export LACLAUGPT_PRIVATE_REPO_ROOT="${LACLAUGPT_PRIVATE_REPO_ROOT:-${SCRATCH_ROOT}/LaclauGPT-Private}"

  export LACLAUGPT_VLLM_TEST_ENV_FILE="${LACLAUGPT_VLLM_TEST_ENV_FILE:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/vllm_video_test.env}"

  if [[ ! -f "${LACLAUGPT_VLLM_TEST_ENV_FILE}" ]]; then
    _roihu_vllm_fail "private settings file missing: ${LACLAUGPT_VLLM_TEST_ENV_FILE}" || return 1
  fi

  # Load the persistent private settings into this login shell.
  set -a
  # shellcheck disable=SC1090
  source "${LACLAUGPT_VLLM_TEST_ENV_FILE}"
  set +a

  # Re-apply dependable defaults after the env file has had a chance to
  # override them.
  export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SCRATCH_ROOT}/LaclauGPT-Multimodal-Analysis}"
  export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-${SCRATCH_ROOT}/laclaugpt-multimodal}"
  export LACLAUGPT_PRIVATE_REPO_ROOT="${LACLAUGPT_PRIVATE_REPO_ROOT:-${SCRATCH_ROOT}/LaclauGPT-Private}"
  export LACLAUGPT_VLLM_TEST_VENV="${LACLAUGPT_VLLM_TEST_VENV:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/.venv-roihu-vllm}"
  export LACLAUGPT_VLLM_TEST_INPUT_CSV="${LACLAUGPT_VLLM_TEST_INPUT_CSV:-${LACLAUGPT_PRIVATE_REPO_ROOT}/analysis/ep24_reprocess/experiments/vllm_video_test_input.csv}"

  export RCLONE_REMOTE="${RCLONE_REMOTE:-s3allas}"
  export ALLAS_BUCKET="${ALLAS_BUCKET:-HEPP24}"

  export HF_HOME="${HF_HOME:-${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/hf-cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"
  export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-72}"
  export OMP_PLACES=cores
  export OMP_PROC_BIND=spread
  export CW_FORCE_CONDA_ACTIVATE=1

  mkdir -p     "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/logs"     "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/csv"     "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/vllm_video_downloads"     "${HF_HOME}"     "${HF_HUB_CACHE}"

  [[ -d "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/.git" ]] || {
    _roihu_vllm_fail "public repo missing: ${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}" || return 1
  }
  [[ -x "${LACLAUGPT_VLLM_TEST_VENV}/bin/python" ]] || {
    _roihu_vllm_fail "vLLM venv missing: ${LACLAUGPT_VLLM_TEST_VENV}" || return 1
  }
  [[ -f "${LACLAUGPT_VLLM_TEST_INPUT_CSV}" ]] || {
    _roihu_vllm_fail "prepared video-test CSV missing: ${LACLAUGPT_VLLM_TEST_INPUT_CSV}" || return 1
  }

  # Clean out a previous reconnect's environment before loading the exact stack.
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate >/dev/null 2>&1 || true
  fi

  module --force purge || return 1
  module load python-vllm || return 1
  module load allas || return 1
  module load gcc/14.3.0 ffmpeg || return 1

  # shellcheck disable=SC1090
  source "${LACLAUGPT_VLLM_TEST_VENV}/bin/activate" || return 1

  unset PYTHONPATH
  unset PYTHONHOME

  export RCLONE_BIN
  RCLONE_BIN="$(command -v rclone || true)"

  command -v ffmpeg >/dev/null 2>&1 || {
    _roihu_vllm_fail "ffmpeg missing after module load gcc/14.3.0 ffmpeg" || return 1
  }
  command -v ffprobe >/dev/null 2>&1 || {
    _roihu_vllm_fail "ffprobe missing after module load gcc/14.3.0 ffmpeg" || return 1
  }
  [[ -n "${RCLONE_BIN}" && -x "${RCLONE_BIN}" ]] || {
    _roihu_vllm_fail "rclone missing after module load allas" || return 1
  }

  python - <<'PY' || return 1
import sys
from pathlib import Path

import transformers
import vllm

version = getattr(transformers, "__version__", "0")
major_text = version.split(".", 1)[0]
major = int(major_text) if major_text.isdigit() else 0

print(f"transformers : {version}")
print(f"transformers : {Path(transformers.__file__).resolve()}")
print(f"vllm         : {getattr(vllm, '__version__', '<unknown>')}")
print(f"vllm path    : {Path(vllm.__file__).resolve()}")

if major < 5:
    sys.stderr.write(
        f"ERROR: vLLM 0.29 requires Transformers 5+, found {version}\n"
    )
    raise SystemExit(2)
PY

  # Allas credentials/configuration normally survives reconnects. Verify it
  # without printing secrets. If this fails, rerun allas-conf once.
  if ! "${RCLONE_BIN}" lsd "${RCLONE_REMOTE}:" >/dev/null 2>&1; then
    _roihu_vllm_fail "Allas remote ${RCLONE_REMOTE}: is not usable. Run: allas-conf ${CSC_PROJECT}" || return 1
  fi

  # Convenience function retained in the shell after sourcing.
  roihu_vllm_submit() {
    (
      cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}/logs" || exit 1
      sbatch \
        --account="${CSC_PROJECT}" \
        "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/scripts/roihu/vllm_video_test.sbatch" \
        "$@"
    )
  }
  export -f roihu_vllm_submit

  if [[ "${ROIHU_VLLM_NO_CD:-0}" != "1" ]]; then
    cd "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}" || return 1
  fi

  printf '\n'
  printf 'Roihu vLLM environment READY\n'
  printf '  project      : %s\n' "${CSC_PROJECT}"
  printf '  public repo  : %s\n' "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}"
  printf '  private root : %s\n' "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
  printf '  input CSV    : %s\n' "${LACLAUGPT_VLLM_TEST_INPUT_CSV}"
  printf '  venv         : %s\n' "${LACLAUGPT_VLLM_TEST_VENV}"
  printf '  Allas        : %s:%s\n' "${RCLONE_REMOTE}" "${ALLAS_BUCKET}"
  printf '  rclone       : %s\n' "${RCLONE_BIN}"
  printf '  ffmpeg       : %s\n' "$(command -v ffmpeg)"
  printf '\nSubmit with: roihu_vllm_submit\n'
}

_roihu_vllm_bootstrap
_bootstrap_status=$?
unset -f _roihu_vllm_bootstrap
unset -f _roihu_vllm_fail
return "${_bootstrap_status}" 2>/dev/null || exit "${_bootstrap_status}"
