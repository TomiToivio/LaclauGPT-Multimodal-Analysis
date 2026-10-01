# vLLM whole-video smoke test (Qwen3-VL-8B) on CSC Roihu

An **isolated experiment**. It does not touch the five-stage EP24 pipeline, the
Ollama backend, or any legacy CSV contract. It answers one question:

> Can we submit a clean sbatch job on CSC Roihu, download five real EP24 videos
> from CSC Allas, analyze the videos directly with Qwen3-VL-8B through vLLM, and
> write readable CSV + debug-log output?

Files:

| Path | Role |
| --- | --- |
| `experiments/vllm_video_test.py` | the standalone test |
| `scripts/roihu/vllm_video_test.sbatch` | the Slurm batch file |
| `docs/VLLM_VIDEO_TEST.md` | this document |

Nothing here is wired into `run_pipeline.sh`. A future issue decides whether
native whole-video analysis joins the production pipeline.

## What it does

1. Reads the EP24 CSV with pandas (`dtype=str`, so IDs are never coerced).
2. Selects **five usable rows** — reproducible for a given `--seed`.
3. Derives each video's Allas object path from the row metadata.
4. Downloads only those five objects into job-local scratch.
5. Loads **`Qwen/Qwen3-VL-8B-Instruct`** through vLLM.
6. Sends each video as **native video input**, one video per request, in
   temporal order.
7. Writes the selected rows plus `vllm_video_*` columns to a new CSV.
8. Keeps the source CSV untouched.
9. Logs a failure and continues when a single video cannot be fetched or
   analyzed.
10. Writes a verbose debug log throughout.

## Object path convention

The repository already fixes the Allas layout, so this test reuses it rather
than inventing a second one (`docs/LEGACY_PIPELINE_CONTRACT.md` §1,
`roihu_preprocess.py`):

```
Allas/Scraper/TikTok/Videos/<scrapedCountry>/<authorUniqueId>/<videoId>.mp4
```

Only the part **inside** the bucket is composed by the script
(`--allas-path-template`); the endpoint and bucket stay in the environment.

## vLLM video-input path (documented as required by §4)

vLLM's offline `LLM.generate` with a `video` modality — not six image prompts:

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct", limit_mm_per_prompt={"video": 1})
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
_, video_inputs, video_kwargs = process_vision_info(
    messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
videos, metadatas = zip(*video_inputs)
llm.generate([{
    "prompt": prompt,
    "multi_modal_data": {"video": list(videos)},
    "mm_processor_kwargs": {**video_kwargs, "video_metadata": list(metadatas)},
}], SamplingParams(max_tokens=2048))
```

The application-level API receives the source as a **video**; frame sampling
happens inside the Qwen preprocessing stack and temporal order is preserved.
This is what makes it a whole-video analysis rather than six stills.

**Qwen3-VL specifics** the code relies on, and which the debug log records from
the real job so a version difference is visible rather than hidden:

- `process_vision_info` needs `image_patch_size=16` and
  `return_video_metadata=True` for the Qwen3-VL series.
- Qwen3-VL-era vLLM expects the per-video metadata to travel as
  `mm_processor_kwargs` on the request.
- The video decode backend is vLLM's default. If the module's default cannot
  decode a given container, set `VLLM_VIDEO_LOADER_BACKEND` (e.g. `torchcodec`)
  or `--media-io-kwargs '{"video": {"backend": ...}}'` and note it here.

CSC documents `python-vllm` with default **0.29.0** and also 0.19.1 / 0.18.0
(<https://docs.csc.fi/apps/vllm>). If the installed version exposes a slightly
different API, **adapt to the observed Roihu version** rather than pinning an
incompatible external vLLM.

## From zero to a test run on Roihu

Assumes you are logged into Roihu and have a private project directory. Replace
`project_200xxxx` with your own allocation.

```bash
# 1. clone or update the public repository
cd "$HOME"
git clone https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis.git
cd LaclauGPT-Multimodal-Analysis && git pull

# 2. go to private project/scratch storage (never HOME for outputs)
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT="$PWD"
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT="/scratch/project_200xxxx/ep24-multimodal"
mkdir -p "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT"/{logs,csv,vllm_video_downloads,.hf_cache}

# 3. load the CSC vLLM module
module --force purge
module load python-vllm

# 4. create the project-local venv for the few extra packages
#    (Roihu is aarch64 — never copy a Puhti x86 venv here)
python -m venv --system-site-packages "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/.venv-roihu-vllm"
source "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/.venv-roihu-vllm/bin/activate"

# 5. install only the missing video/Qwen dependencies, pinned and minimal
pip install "qwen-vl-utils>=0.0.14" "av>=13,<16"
#    Add a decoder only if the default backend fails on your container, e.g.
#    pip install "torchcodec>=0.2"   # and set VLLM_VIDEO_LOADER_BACKEND=torchcodec

# 6. keep model downloads on project scratch, not HOME
export HF_HOME="$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"

# 7. configure Allas (S3-compatible) for rclone; credentials stay private
export RCLONE_CONFIG="$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/rclone.conf"
export ALLAS_BUCKET="<your-bucket>"
rclone lsd allas:            # sanity check the remote

# 8. point at the private EP24 input CSV
export LACLAUGPT_VLLM_TEST_INPUT_CSV="$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/csv/tiktok_videos.csv"

# 9. tiny sanity check that the module and extras import
python -c "import vllm, torch, transformers, qwen_vl_utils; \
print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 10. submit
sbatch "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/scripts/roihu/vllm_video_test.sbatch"

# 11. check the queue
squeue -u "$USER"

# 12. follow Slurm stdout/stderr
tail -f vllm-video-test-*.out vllm-video-test-*.err

# 13. follow the Python debug log (the detailed one)
tail -f "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT"/logs/vllm_video_test_*.log

# 14. find the result CSV when the job finishes
ls -l "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT"/csv/ep24_vllm_video_test_*.csv
```

### Rerunning with a fixed seed

Selection is deterministic, so a fixed seed reproduces the same five videos:

```bash
export LACLAUGPT_VLLM_TEST_SEED=20261001
sbatch scripts/roihu/vllm_video_test.sbatch
```

### Testing the harness without a GPU

Useful before spending a GPU hour, and the way this script is verified in CI:

```bash
python experiments/vllm_video_test.py \
  --input-csv tests/fixtures/ep24_vllm_test_sample.csv \
  --output-csv /tmp/out.csv --log-path /tmp/out.log \
  --fetch-backend local --allas-local-root tests/fixtures/allas \
  --model-backend stub --sample-size 5
```

`--model-backend stub` performs **no** inference and says so in the output
column. It exercises selection, path derivation, fetch, logging, the CSV
contract and per-row failure isolation.

## Configuration

Everything is a CLI flag with an environment-variable default, so the sbatch
file stays thin and nothing private is committed.

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--input-csv` | `LACLAUGPT_VLLM_TEST_INPUT_CSV` | required |
| `--output-csv` | `LACLAUGPT_VLLM_TEST_OUTPUT_CSV` | `<input>_vllm_test.csv` |
| `--log-path` | `LACLAUGPT_VLLM_TEST_LOG` | `./logs/vllm_video_test.log` |
| `--download-dir` | `LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR` | `./vllm_video_downloads` |
| `--model` | `LACLAUGPT_VLLM_TEST_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` |
| `--sample-size` | `LACLAUGPT_VLLM_TEST_SAMPLE_SIZE` | `5` |
| `--seed` | `LACLAUGPT_VLLM_TEST_SEED` | `20261001` |
| `--allas-bucket` | `ALLAS_BUCKET` | none |
| `--fetch-backend` | `LACLAUGPT_VLLM_TEST_FETCH_BACKEND` | `rclone` |
| `--rclone-remote` | `RCLONE_REMOTE` | `allas` |

The model default is `Qwen/Qwen3-VL-8B-Instruct` and the script will **not**
silently substitute another model.

## Output columns

The selected source rows keep every original column; these are appended:

`vllm_video_model`, `vllm_video_status`, `vllm_video_analysis`,
`vllm_video_error`, `vllm_video_remote_path`, `vllm_video_local_path`,
`vllm_video_bytes`, `vllm_video_runtime_seconds`, `vllm_video_selected_index`,
`vllm_video_prompt`.

`vllm_video_status` is `ok` or `error`; a failed row carries the exception text
in `vllm_video_error` and the run continues.

## Debug log

The log records UTC and local time, hostname, SLURM job ID, GPU (`nvidia-smi`),
Python/vLLM/torch/CUDA versions, all paths, the seed, each selected row's
identifiers, the remote and local path, file size, `ffprobe` metadata, model
initialisation, per-video start/end and runtime, the prompt, the raw response,
exceptions with tracebacks, and the final success/failure counts. It is meant to
be verbose — this is a test harness.

## Privacy

No credentials, endpoints, bucket names, allocation IDs or private paths are
committed. The EP24 CSV is real research data and stays in private storage:
`tests/fixtures/` contains only a synthetic CSV with fake IDs for harness tests.
The test never writes to Allas and never modifies the source CSV.

## Roihu caveats to confirm on the first real run

These are the assumptions most likely to differ from reality. The debug log is
designed to answer each one from the job output:

1. **Which `python-vllm` version is loaded** and whether it is the default 0.29.0.
2. **Whether `process_vision_info` needs `image_patch_size=16`** and
   `return_video_metadata=True` on that version, and the exact key names it returns.
3. **Whether the module's video decode backend** reads the EP2024 containers; if
   not, which backend (`torchcodec`, PyAV, decord) fixes it.
4. **Whether one GH200 and the pixel budget** used here fit the chosen frame
   count, or whether `--video-total-pixels` / `fps` must be lowered.
5. **Whether the model is already cached** in project scratch or must be
   downloaded on first run (a large one-off download).


## EP24 initial-scroll and splitter-failure rule

This experiment now follows the repository-wide EP24 video contract documented
in EP24_VIDEO_SCROLL_ARTIFACTS.md. After downloading the unchanged source clip
from Allas, the harness creates a derived analysis clip beginning at original
t=1.0s and sends that derived clip to vLLM.

The model is instructed not to count the known initial splitter artifact. It
checks the remaining video for later TikTok/Instagram feed transitions and
appends structured SCROLL and SCROLL_SECONDS metadata while retaining the
human-readable descriptive analysis. The CSV also includes needs_resplit and
video_initial_skip_seconds.

SCROLL_SECONDS uses timestamps on the original source timeline. A detected
additional scroll is queued for the deterministic, depth-limited re-split
workflow rather than destructively editing the source on first-pass model output.
