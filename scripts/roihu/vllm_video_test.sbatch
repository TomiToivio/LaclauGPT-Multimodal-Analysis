#!/usr/bin/env bash
#SBATCH --job-name=laclaugpt-vllm-video
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --output=vllm_video_test_%j.out
#SBATCH --error=vllm_video_test_%j.err

set -euo pipefail

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
: "${LACLAUGPT_VLLM_INPUT_CSV:?Set LACLAUGPT_VLLM_INPUT_CSV to the private EP24 CSV}"
: "${LACLAUGPT_VLLM_WORK_ROOT:?Set LACLAUGPT_VLLM_WORK_ROOT to project scratch}"

WORK_ROOT=${LACLAUGPT_VLLM_WORK_ROOT}
VENV=${LACLAUGPT_VLLM_VENV:-${WORK_ROOT}/.venv-vllm-video}
MODEL=${LACLAUGPT_VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
SEED=${LACLAUGPT_VLLM_SEED:-19}
SAMPLE_SIZE=${LACLAUGPT_VLLM_SAMPLE_SIZE:-5}
FPS=${LACLAUGPT_VLLM_FPS:-1.0}
ALLAS_ROOT=${LACLAUGPT_VLLM_ALLAS_ROOT:-s3allas:}
REMOTE_COLUMN=${LACLAUGPT_VLLM_VIDEO_REMOTE_COLUMN:-}
REMOTE_TEMPLATE=${LACLAUGPT_VLLM_VIDEO_REMOTE_TEMPLATE:-}

JOB_DIR=${WORK_ROOT}/vllm-video-test/${SLURM_JOB_ID}
TMP_VIDEO_DIR=${JOB_DIR}/videos
OUTPUT_CSV=${LACLAUGPT_VLLM_OUTPUT_CSV:-${JOB_DIR}/vllm_video_results.csv}
DEBUG_LOG=${LACLAUGPT_VLLM_DEBUG_LOG:-${JOB_DIR}/vllm_video_debug.log}

mkdir -p "${JOB_DIR}" "${TMP_VIDEO_DIR}" "${WORK_ROOT}/hf-cache"

module --force purge
module load python-vllm
module load allas
module load ffmpeg
unset PYTHONPATH
unset PYTHONHOME

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing project-local venv: ${VENV}" >&2
  echo "Create it on roihu-gpu.csc.fi with --system-site-packages; see docs/VLLM_VIDEO_TEST.md" >&2
  exit 2
fi
source "${VENV}/bin/activate"

export HF_HOME=${HF_HOME:-${WORK_ROOT}/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TMPDIR=${TMPDIR:-${JOB_DIR}/tmp}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
mkdir -p "${TMPDIR}" "${HUGGINGFACE_HUB_CACHE}"

command -v rclone >/dev/null 2>&1 || {
  echo "rclone not found after 'module load allas'" >&2
  exit 2
}
command -v ffprobe >/dev/null 2>&1 || {
  echo "ffprobe not found after 'module load ffmpeg'" >&2
  exit 2
}

python - <<'PY'
import importlib
required = ["pandas", "qwen_vl_utils", "transformers", "vllm"]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'version n/a')}")
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n" + "\n".join(missing))
PY

echo "Checking Allas rclone configuration..."
rclone lsd "${ALLAS_ROOT%%/*}" >/dev/null || {
  echo "Cannot access Allas remote via rclone: ${ALLAS_ROOT}" >&2
  echo "Run 'module load allas' and 'allas-conf' on roihu-gpu.csc.fi before submitting." >&2
  exit 3
}

echo "Job directory: ${JOB_DIR}"
echo "Result CSV: ${OUTPUT_CSV}"
echo "Debug log: ${DEBUG_LOG}"
echo "SLURM stdout: vllm_video_test_${SLURM_JOB_ID}.out"
echo "SLURM stderr: vllm_video_test_${SLURM_JOB_ID}.err"

ARGS=(
  --input-csv "${LACLAUGPT_VLLM_INPUT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --log "${DEBUG_LOG}"
  --tmp-dir "${TMP_VIDEO_DIR}"
  --model "${MODEL}"
  --sample-size "${SAMPLE_SIZE}"
  --seed "${SEED}"
  --fps "${FPS}"
  --allas-root "${ALLAS_ROOT}"
)

if [[ -n "${REMOTE_COLUMN}" ]]; then
  ARGS+=(--remote-column "${REMOTE_COLUMN}")
fi
if [[ -n "${REMOTE_TEMPLATE}" ]]; then
  ARGS+=(--remote-template "${REMOTE_TEMPLATE}")
fi
if [[ "${LACLAUGPT_VLLM_KEEP_VIDEOS:-0}" == "1" ]]; then
  ARGS+=(--keep-videos)
fi

cd "${PUBLIC_ROOT}"
srun --ntasks=1 python experiments/vllm_video_test.py "${ARGS[@]}"
