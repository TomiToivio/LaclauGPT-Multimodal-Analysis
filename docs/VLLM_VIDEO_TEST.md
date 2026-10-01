# Standalone Qwen3-VL vLLM video smoke test on CSC Roihu

This experiment answers one narrow question: can Roihu download five real EP24 videos from CSC Allas and analyze each **whole video** with `Qwen/Qwen3-VL-8B-Instruct` through vLLM?

It is deliberately isolated from the production EP24 pipeline. It does not replace or modify `roihu_frame.py`, `roihu_summary.py`, the legacy-compatible five-stage flow, Ollama, MongoDB, memory, RDF, DNA, SNA, or `scripts/roihu/run_pipeline.sh`.

## What the experiment does

`experiments/vllm_video_test.py`:

1. reads the real EP24 CSV with Pandas;
2. resolves a video object for each usable row;
3. selects exactly five rows by default with a reproducible random seed;
4. downloads only those selected videos from Allas with `rclone copyto`;
5. passes each local source video through Qwen3-VL's video preprocessing and vLLM's multimodal `video` input;
6. writes the selected legacy rows plus `vllm_video_*` fields to a new CSV;
7. checkpoints the CSV after every video;
8. logs environment details, remote/local paths, video metadata, prompt, raw response, runtime, and tracebacks;
9. continues after an individual download or inference failure.

The input CSV is never changed.

The default model is exactly:

```text
Qwen/Qwen3-VL-8B-Instruct
```

The application treats the input as one temporally ordered video. It does **not** call the model on six unrelated JPEG frames. Internally, Qwen/vLLM may sample frames according to their video preprocessing implementation.

## Current Roihu assumptions

These instructions target **Roihu-GPU**. CSC currently documents full Nvidia GH200 allocation in `gpumedium` with:

```text
--gres=gpu:gh200:1
```

and 72 CPU cores per one-GPU allocation. The batch file follows that layout.

CSC provides vLLM as the `python-vllm` module. Do not install a second pip vLLM into the project environment unless CSC's module is proven incompatible. The project venv is created with `--system-site-packages` so it inherits CSC's vLLM stack and only adds small missing Python dependencies.

CSC's current Roihu Allas module defaults to S3-style access. The experiment uses the `rclone` command supplied by `module load allas`.

## 1. Log in to the correct Roihu login node

GPU software on Roihu is ARM64, so create the environment on the GPU login node:

```bash
ssh roihu-gpu.csc.fi
```

Do not reuse an old Puhti/x86 Python environment.

## 2. Clone or update this repository in project scratch

Replace `project_200xxxx` with your own CSC project.

```bash
mkdir -p /scratch/project_200xxxx
cd /scratch/project_200xxxx

git clone https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis.git
cd LaclauGPT-Multimodal-Analysis
git pull
```

If the repository already exists:

```bash
cd /scratch/project_200xxxx/LaclauGPT-Multimodal-Analysis
git pull
```

## 3. Load CSC vLLM and create a tiny project-local venv

```bash
module --force purge
module load python-vllm

export LACLAUGPT_VLLM_WORK_ROOT=/scratch/project_200xxxx/laclaugpt-vllm-video
mkdir -p "$LACLAUGPT_VLLM_WORK_ROOT"

python3 -m venv --system-site-packages "$LACLAUGPT_VLLM_WORK_ROOT/.venv-vllm-video"
source "$LACLAUGPT_VLLM_WORK_ROOT/.venv-vllm-video/bin/activate"

python -m pip install --upgrade pip
python -m pip install "qwen-vl-utils>=0.0.14" pandas
```

The Qwen3-VL upstream vLLM recipe requires `qwen-vl-utils>=0.0.14` for its video preprocessing path. If CSC's `python-vllm` module already contains a compatible package, pip will normally leave the working dependency in place.

Do **not** run `pip install vllm` here by default. The point is to use CSC's supported vLLM build.

## 4. Configure Hugging Face/model cache in project scratch

```bash
export HF_HOME="$LACLAUGPT_VLLM_WORK_ROOT/hf-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
```

If the model requires authentication in the future, configure Hugging Face credentials privately. Never commit tokens.

## 5. Configure Allas

Load CSC's Allas tools:

```bash
module load allas
```

Configure your Allas connection:

```bash
allas-conf
```

Then verify it:

```bash
rclone lsd s3allas:
```

If your old EP24 objects require Swift access instead, CSC supports:

```bash
allas-conf --swift
rclone lsd allas:
```

and you can later set `LACLAUGPT_VLLM_ALLAS_ROOT` to an `allas:` path.

Do not put Allas credentials in this repository.

## 6. Point the job at the private EP24 CSV

For example:

```bash
export LACLAUGPT_VLLM_INPUT_CSV=/scratch/project_200xxxx/private/ep24_reprocess/data/tiktok_fi.csv
```

The experiment preserves all columns in the five selected rows. Fields such as `authorUniqueId`, `videoId`, `scrapedCountry`, `language`, and `videoDescription` remain untouched.

## 7. Tell the test how EP24 rows map to Allas objects

The script first looks for an explicit video-object column with one of these names:

```text
allas_path
allasPath
video_path
videoPath
video_file
videoFile
object_path
objectPath
```

If your CSV already has one of those fields, set only the Allas root if needed:

```bash
export LACLAUGPT_VLLM_ALLAS_ROOT='s3allas:YOUR_BUCKET'
```

If the real remote path is stored in another CSV column:

```bash
export LACLAUGPT_VLLM_VIDEO_REMOTE_COLUMN='yourActualVideoPathColumn'
export LACLAUGPT_VLLM_ALLAS_ROOT='s3allas:YOUR_BUCKET'
```

If the object path must be constructed from EP24 metadata, use a format template. Example only:

```bash
export LACLAUGPT_VLLM_ALLAS_ROOT='s3allas:YOUR_BUCKET'
export LACLAUGPT_VLLM_VIDEO_REMOTE_TEMPLATE='ep24/{scrapedCountry}/{authorUniqueId}/{videoId}.mp4'
```

The example is **not** a claim about the private EP24 bucket layout. Use the actual layout from private data/configuration.

Rows without a resolvable remote video are excluded before sampling. The job refuses to start inference unless it can select exactly five usable rows by default.

## 8. Optional import/version sanity test

On the GPU login node:

```bash
module --force purge
module load python-vllm
source "$LACLAUGPT_VLLM_WORK_ROOT/.venv-vllm-video/bin/activate"

python - <<'PY'
import pandas
import qwen_vl_utils
import transformers
import vllm

print("pandas", pandas.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("qwen_vl_utils import OK")
PY
```

This is only an import test. Do not load the 8B model on the login node.

## 9. Submit the job

From the public repository checkout:

```bash
cd /scratch/project_200xxxx/LaclauGPT-Multimodal-Analysis
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="$PWD"
```

Then submit:

```bash
sbatch scripts/roihu/vllm_video_test.sbatch
```

If your CSC setup requires an explicit account and no default account is configured, use:

```bash
sbatch --account=project_200xxxx scripts/roihu/vllm_video_test.sbatch
```

The batch file requests one full GH200 in `gpumedium`.

## 10. Check the queue

```bash
squeue -u "$USER"
```

## 11. Follow SLURM stdout/stderr

From the submit directory, replace `JOBID`:

```bash
tail -f "vllm_video_test_JOBID.out"
```

and in another shell if needed:

```bash
tail -f "vllm_video_test_JOBID.err"
```

The batch output prints the exact result CSV and Python debug-log paths near startup.

## 12. Follow the Python debug log

By default the job writes under:

```text
$LACLAUGPT_VLLM_WORK_ROOT/vllm-video-test/$SLURM_JOB_ID/
```

For example:

```bash
tail -f "$LACLAUGPT_VLLM_WORK_ROOT/vllm-video-test/JOBID/vllm_video_debug.log"
```

The log includes:

- hostname and SLURM job ID;
- Python, PyTorch, CUDA, vLLM and GPU information where available;
- input/output paths, seed and model;
- selected EP24 identifiers;
- Allas remote and local paths;
- downloaded file size and `ffprobe` metadata;
- model initialization;
- the full descriptive prompt;
- Qwen video-processing kwargs;
- raw model response;
- per-video runtime;
- full traceback on errors;
- final success/failure counts.

## 13. Locate the result CSV

Default:

```text
$LACLAUGPT_VLLM_WORK_ROOT/vllm-video-test/$SLURM_JOB_ID/vllm_video_results.csv
```

The output contains the selected source rows plus:

```text
vllm_video_model
vllm_video_analysis
vllm_video_status
vllm_video_error
vllm_video_remote_path
vllm_video_local_path
vllm_video_runtime_seconds
vllm_video_metadata
```

A failed video has `vllm_video_status=error`; later videos are still attempted.

## 14. Reproduce the same five-row sample

Set a fixed seed before submission:

```bash
export LACLAUGPT_VLLM_SEED=42
sbatch scripts/roihu/vllm_video_test.sbatch
```

With the same input CSV, remote-resolution settings, sample size, and seed, Pandas selects the same rows.

The default sample size is five. You can override it for debugging:

```bash
export LACLAUGPT_VLLM_SAMPLE_SIZE=5
```

Issue #19's acceptance target remains five videos.

## 15. Useful optional controls

Keep downloaded videos after inference:

```bash
export LACLAUGPT_VLLM_KEEP_VIDEOS=1
```

Change temporal sampling passed to Qwen video preprocessing:

```bash
export LACLAUGPT_VLLM_FPS=1.0
```

Override result/log paths:

```bash
export LACLAUGPT_VLLM_OUTPUT_CSV=/scratch/project_200xxxx/results/vllm_video_results.csv
export LACLAUGPT_VLLM_DEBUG_LOG=/scratch/project_200xxxx/logs/vllm_video_debug.log
```

Override the model only when deliberately testing another model:

```bash
export LACLAUGPT_VLLM_MODEL='Qwen/Qwen3-VL-8B-Instruct'
```

The script never silently substitutes a different default model.

## Known caveats for the first Roihu run

- CSC's current `python-vllm` module/container and Qwen3-VL move faster than the production pipeline. Record the exact versions printed by the debug log.
- The smoke test follows Qwen's direct vLLM recipe: chat template + `process_vision_info(... return_video_kwargs=True, return_video_metadata=True)` + vLLM `multi_modal_data["video"]`. If CSC's installed vLLM exposes a version-specific incompatibility, document the observed error before changing the approach.
- Video decoding depends on codecs present in the Roihu environment. `ffprobe` metadata is logged to make decode failures inspectable.
- Long/high-resolution videos can create large multimodal contexts. The default `fps=1.0` is intentionally conservative for a five-video capability test.
- The model itself may internally sample/compress frames. That is still native ordered-video processing at the application level, not six independent still-image analyses.
- Allas authentication must already be configured for the batch environment. The sbatch script validates the rclone remote and fails early with a readable message when it cannot connect.

## Files added by this experiment

```text
experiments/vllm_video_test.py
scripts/roihu/vllm_video_test.sbatch
docs/VLLM_VIDEO_TEST.md
```

Nothing here is wired into the production runner.
