# EP24 native-video vLLM test on CSC Roihu

This document is the runbook for issue #24. The experiment is intentionally isolated from the production EP24 pipeline. It downloads the prepared private sample from CSC Allas, preserves the source file, removes the first **1.0 second** into a derived analysis clip, sends only that derived clip to vLLM, and writes a human-readable CSV plus exhaustive debug logs.

The normal target is the private file:

```
LaclauGPT-Private/analysis/ep24_reprocess/experiments/vllm_video_test_input.csv
```

That CSV should contain about 10 reproducibly selected usable videos. The public repository must never contain the real private sample or video binaries.

## Files

| File | Purpose |
| --- | --- |
| `experiments/vllm_video_test.py` | isolated native-video experiment harness |
| `experiments/requirements-vllm-video-test.txt` | extra packages layered on CSC's `python-vllm` module |
| `config/vllm_video_test.env.example` | public non-secret environment/settings template |
| `scripts/roihu/vllm_video_test.sbatch` | one-GH200 Roihu batch job |
| `tests/test_vllm_video_test_harness.py` | synthetic CPU/stub tests |
| `docs/EP24_VIDEO_SCROLL_ARTIFACTS.md` | repository-wide first-second/scroll contract |

## Important Roihu differences from the old A100/Ollama example

Do not copy the old Puhti/Mahti-style A100/Ollama resource block into Roihu.

Current Roihu GPU nodes use NVIDIA GH200 accelerators. For one full GPU, CSC documents:

```bash
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
```

The test uses CSC's `python-vllm` module directly. It does **not** start `ollama serve`, pull an Ollama model, or load `python-pytorch` separately.

The allocation/project is intentionally not hard-coded in the script. Supply it when submitting:

```bash
sbatch --account="$CSC_PROJECT" scripts/roihu/vllm_video_test.sbatch
```

CSC references:

- https://docs.csc.fi/computing/running/creating-job-scripts-roihu/
- https://docs.csc.fi/computing/running/example-job-scripts-roihu/
- https://docs.csc.fi/apps/pytorch/
- https://docs.csc.fi/computing/allas-in-roihu/

## 1. Log in to Roihu GPU

Use the Roihu GPU login node according to your CSC setup.

Create variables for your CSC project and project scratch. Replace the example project ID once, in your shell only:

```bash
export CSC_PROJECT=project_2009497
export SCRATCH_ROOT="/scratch/$CSC_PROJECT"
```

Nothing in the repository hard-codes this value.

## 2. Clone or update the public repository

```bash
cd "$SCRATCH_ROOT"
if [ ! -d LaclauGPT-Multimodal-Analysis/.git ]; then
  git clone https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis.git
fi
cd LaclauGPT-Multimodal-Analysis
git pull --ff-only
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="$PWD"
```

For the issue #24 development branch before it is merged:

```bash
git fetch origin
git switch roihu-vllm-video-10pack
git pull --ff-only
```

## 3. Clone or update the private repository

Keep the real EP24 data/settings in private storage:

```bash
cd "$SCRATCH_ROOT"
if [ ! -d LaclauGPT-Private/.git ]; then
  git clone git@github.com:TomiToivio/LaclauGPT-Private.git
fi
cd LaclauGPT-Private
git pull --ff-only
export LACLAUGPT_PRIVATE_REPO_ROOT="$PWD"
```

Expected prepared sample:

```bash
export LACLAUGPT_VLLM_TEST_INPUT_CSV="$LACLAUGPT_PRIVATE_REPO_ROOT/analysis/ep24_reprocess/experiments/vllm_video_test_input.csv"
test -f "$LACLAUGPT_VLLM_TEST_INPUT_CSV"
```

Inspect the sample without dumping sensitive content into job logs:

```bash
python - "$LACLAUGPT_VLLM_TEST_INPUT_CSV" <<'PY'
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p, dtype=str, keep_default_na=False)
print("rows:", len(df))
print("columns:", list(df.columns))
print("allas_filename non-empty:", int(df.get("allas_filename", "").astype(bool).sum()) if "allas_filename" in df else "column absent")
PY
```

The intended real run is approximately 10 rows. The sbatch wrapper processes the complete prepared sample and does not intentionally draw a second random subset.

## 4. Create the private runtime area

```bash
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="$SCRATCH_ROOT/laclaugpt-multimodal"
mkdir -p "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT"/{logs,csv,vllm_video_downloads,hf-cache}
```

Copy the public settings template to private storage:

```bash
cp "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/config/vllm_video_test.env.example" \
   "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/vllm_video_test.env"
chmod 600 "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/vllm_video_test.env"
```

Edit the private copy. At minimum set:

- `LACLAUGPT_MULTIMODAL_PUBLIC_ROOT`
- `LACLAUGPT_MULTIMODAL_PRIVATE_ROOT`
- `LACLAUGPT_VLLM_TEST_INPUT_CSV`
- `LACLAUGPT_VLLM_TEST_VENV`
- `ALLAS_BUCKET`

Then load it:

```bash
set -a
source "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/vllm_video_test.env"
set +a
```

Do not store passwords, tokens, access keys, MongoDB URIs, Redis URLs or other secrets in the public template.

## 5. Configure CSC Allas

On Roihu, load the Allas tools:

```bash
module load allas
```

Configure S3 access interactively:

```bash
allas-conf "$CSC_PROJECT"
```

Or, if the Allas project differs from the compute allocation:

```bash
allas-conf project_XXXXXXXX
```

Current Roihu `allas-conf` defaults to S3. It creates `s3allas:` and a project-specific `s3allas-project_...:` rclone remote. The public settings template therefore defaults to:

```bash
export RCLONE_REMOTE=s3allas
```

Verify access:

```bash
check-allas-connections
rclone lsd "$RCLONE_REMOTE:"
rclone lsf "$RCLONE_REMOTE:$ALLAS_BUCKET" | head
```

Do not print the contents of rclone credential files into logs.

If the historical media were stored through Swift and must be accessed using the Swift endpoint instead, configure it explicitly:

```bash
module load allas
allas-conf --swift
export RCLONE_REMOTE=allas
```

Use one protocol consistently for those objects.

## 6. Load vLLM and build the project-local environment

Load the CSC vLLM module first:

```bash
module --force purge
module load python-vllm
```

Create a project-scratch virtual environment that inherits CSC's GPU stack:

```bash
export LACLAUGPT_VLLM_TEST_VENV="$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/.venv-roihu-vllm"

python -m venv --system-site-packages "$LACLAUGPT_VLLM_TEST_VENV"
source "$LACLAUGPT_VLLM_TEST_VENV/bin/activate"
python -m pip install -U pip
python -m pip install -r "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/experiments/requirements-vllm-video-test.txt"
```

Do **not** `pip install vllm` or `pip install torch` into this environment. The experiment requirements intentionally layer only the missing Python packages over CSC's `python-vllm` module.

Check imports and versions:

```bash
python - <<'PY'
import torch, transformers, vllm, qwen_vl_utils, pandas, av
print("vllm:", getattr(vllm, "__version__", "?"))
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("qwen-vl-utils:", getattr(qwen_vl_utils, "__version__", "?"))
print("pandas:", pandas.__version__)
print("av:", av.__version__)
PY
```

The Python harness logs the observed runtime versions again inside the batch job. If the CSC module changes, trust the observed Roihu version rather than assuming a version from an older document.

## 7. Keep Hugging Face caches off HOME

The settings/batch file default to:

```bash
export HF_HOME="$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

The first real run may need to populate the model cache.

## 8. Mandatory EP24 first-second trim

For every real EP24 video:

1. download the original source unchanged;
2. probe it;
3. create a separate derived analysis clip beginning at source `t=1.0s`;
4. probe the derived clip;
5. give only the derived clip to vLLM;
6. retain `video_initial_skip_seconds = 1.0` in provenance.

The first second is the tail of the preceding feed item caused by the scroll-based splitter. It is not optional preprocessing.

Never destructively overwrite the downloaded source video.

If vLLM detects an additional later feed scroll, the model output records `SCROLL` and `SCROLL_SECONDS`; that is evidence for later re-splitting, not permission to mutate the source during this experiment.

## 9. Researcher prompt

The predefined human-researcher questions must remain easy to edit manually.

The current harness keeps the researcher-facing text in the clearly marked constants near the top of:

```
experiments/vllm_video_test.py
```

Do not scatter the final research questions across helper functions. Before a publication-quality run, record the exact prompt version/hash in the output provenance. Issue #24 treats structured JSON as optional; the raw and human-readable answer remains mandatory.

## 10. CPU/stub dry run before spending GPU time

Run the synthetic tests:

```bash
cd "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT"
source "$LACLAUGPT_VLLM_TEST_VENV/bin/activate"
pytest -q tests/test_vllm_video_test_harness.py tests/test_ep24_video_scroll.py
```

You can also run the harness against synthetic fixtures with the stub model. This performs no real inference:

```bash
python experiments/vllm_video_test.py \
  --input-csv tests/fixtures/ep24_vllm_test_sample.csv \
  --output-csv /tmp/ep24-vllm-stub.csv \
  --log-path /tmp/ep24-vllm-stub.log \
  --download-dir /tmp/ep24-vllm-downloads \
  --fetch-backend local \
  --allas-local-root tests/fixtures/allas \
  --model-backend stub \
  --sample-size 5 \
  --seed 20261001
```

The stub path verifies control flow, CSV writing, failure isolation and logging. It is not a video-model result.

## 11. Submit the real ~10-video job

Reload the private settings in the shell you submit from:

```bash
set -a
source "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/vllm_video_test.env"
set +a
```

Submit from the private log directory so the Slurm stdout/stderr files also land in private project storage:

```bash
cd "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/logs"

sbatch \
  --account="$CSC_PROJECT" \
  "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/scripts/roihu/vllm_video_test.sbatch"
```

The script requests:

- one node;
- one task;
- 72 CPU cores;
- one GH200;
- `gpumedium`;
- two hours.

Those values are intentionally conservative for a first 10-video vLLM experiment. Tune only after seeing the real runtime/memory diagnostics.

## 12. Monitor the job

Queue:

```bash
squeue -u "$USER"
```

Detailed Slurm information:

```bash
scontrol show job <JOBID>
```

Follow Slurm output:

```bash
tail -f "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/logs/vllm-video-test-<JOBID>.out"
```

Follow the deliberately verbose Python log:

```bash
tail -f "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/logs/vllm_video_test_<JOBID>.log"
```

GPU view while allocated:

```bash
srun --jobid=<JOBID> nvidia-smi
```

## 13. Locate and inspect the output

Default result:

```bash
ls -lh "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/csv/ep24_vllm_video_test_"*.csv
```

Human-readable inspection:

```bash
python - "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/csv/ep24_vllm_video_test_<JOBID>.csv" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
cols = [c for c in [
    "videoId", "authorUniqueId", "scrapedCountry",
    "vllm_video_status", "video_initial_skip_seconds",
    "SCROLL", "SCROLL_SECONDS", "vllm_video_analysis", "vllm_video_error"
] if c in df.columns]
print(df[cols].to_string(index=False))
PY
```

The output must be additive: input metadata plus experiment/provenance/result fields. The private source dataframe is never overwritten.

## 14. Reproduce exactly the same 10-video sample

The reproducible sample selection happens when the private `vllm_video_test_input.csv` is created. Preserve:

- source CSV path/name;
- fixed seed;
- selected stable identifiers;
- exact Allas object field/path;
- prompt version/hash;
- model identifier;
- code commit SHA.

For reruns, use the same prepared sample file rather than selecting 10 new videos from the full EP24 dataframe.

Record the public code SHA:

```bash
git -C "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT" rev-parse HEAD
```

Record a checksum of the private sample without exposing its content:

```bash
sha256sum "$LACLAUGPT_VLLM_TEST_INPUT_CSV"
```

## 15. Troubleshooting

### `rclone: command not found`

Load Allas:

```bash
module load allas
```

### `didn't find section in config file` or remote missing

Check:

```bash
module load allas
check-allas-connections
rclone listremotes
```

Run `allas-conf` again if needed.

### S3 remote vs Swift remote

On Roihu, normal `allas-conf` creates an S3 remote named `s3allas:`. `allas-conf --swift` creates/uses `allas:` for Swift. Make `RCLONE_REMOTE` match the protocol used for the original objects.

### Object not found

Check the exact input field such as `allas_filename` and bucket. Do not silently invent a replacement key. The experiment should retain compatibility with the historical derived path convention only as a fallback.

### `ffmpeg` / `ffprobe` missing

The real run must not bypass the first-second rule. If these commands are unavailable in the loaded module environment, add the appropriate CSC module/tooling before running. Do not send the untrimmed source to vLLM.

### Video decoder error

First inspect:

```bash
ffprobe -hide_banner "/path/to/source.mp4"
ffprobe -hide_banner "/path/to/trimmed.mp4"
```

If ffmpeg can decode it but the Qwen/vLLM media path cannot, test a supported loader backend and record the change. The public env template leaves an optional `VLLM_VIDEO_LOADER_BACKEND` override commented out for this reason.

### CUDA out of memory

Reduce video budget first:

- `LACLAUGPT_VLLM_TEST_VIDEO_TOTAL_PIXELS`
- `LACLAUGPT_VLLM_TEST_VIDEO_MAX_PIXELS`
- `LACLAUGPT_VLLM_TEST_MAX_MODEL_LEN`
- `LACLAUGPT_VLLM_TEST_MAX_TOKENS`

Do not silently switch models in the middle of a reproducibility run.

### Structured output unsupported

Structured decoding is optional for this experiment. Save the complete raw natural-language response, keep the human-readable analysis column, mark structuring as unsupported/failed, and postprocess later.

### One video fails

The harness is designed to isolate per-video failures. The other rows should still complete and the failed row should carry a readable error plus traceback in the debug log.

## 16. What the debug log should capture

Useful diagnostics include:

- UTC and local timestamps;
- hostname and Slurm identifiers;
- Python/vLLM/PyTorch/Transformers/qwen-vl-utils versions;
- GPU model, memory and driver;
- non-secret configuration;
- input/output/work paths;
- stable row/video identifiers;
- Allas object/path used;
- download backend and result;
- byte count/checksum where implemented;
- ffprobe metadata before/after trim;
- exact trim operation;
- model/pixel/token budgets;
- complete prompt;
- inference runtime;
- raw model answer;
- structured parse status;
- exception tracebacks;
- cleanup;
- final success/failure summary.

Never log credentials, tokens, complete environment dumps, MongoDB/Redis secrets, private authentication URLs or the contents of rclone credential files.

## 17. Minimal copy/paste recipe

After one-time cloning, private settings creation, `allas-conf`, and venv installation:

```bash
export CSC_PROJECT=project_2009497
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="/scratch/$CSC_PROJECT/laclaugpt-multimodal"

module --force purge
module load allas
module load python-vllm

set -a
source "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/vllm_video_test.env"
set +a

source "$LACLAUGPT_VLLM_TEST_VENV/bin/activate"

check-allas-connections
rclone lsd "$RCLONE_REMOTE:"

cd "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/logs"
sbatch --account="$CSC_PROJECT" \
  "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/scripts/roihu/vllm_video_test.sbatch"
```

Then:

```bash
squeue -u "$USER"
```

and inspect the final CSV under:

```
$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/csv/
```


## Laskin fallback experiment

For the measured Volta compatibility result, pinned fallback environment, and non-Slurm wrapper, see [LASKIN_VLLM_VIDEO_TEST.md](LASKIN_VLLM_VIDEO_TEST.md).
