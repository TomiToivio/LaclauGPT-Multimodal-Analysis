# Roihu runbook — EP24 pipeline steps 1–6

Audience: whoever submits the first real end-to-end runs. Issue:
[TomiToivio/LaclauGPT-Multimodal-Analysis#188](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/issues/188).

Steps 7–9 are optional/experimental and out of scope here.

## Two logins, and why it matters

Roihu has **separate login nodes per architecture**:

- `roihu-gpu.csc.fi` — ARM64 (NVIDIA Grace / GH200). All of steps 1–6.
- `roihu-cpu.csc.fi` — x86 (AMD).

Install **and** submit from a GPU login node. `install_roihu.sh` checks this and
refuses to run on a non-`aarch64` host (override with
`LACLAUGPT_ALLOW_NON_ROIHU_INSTALL=1` only for a deliberate local test). A venv
built on one side does not work on the other, and that failure surfaces as a
confusing import error rather than an architecture mismatch.

## One-time setup

```bash
ssh roihu-gpu.csc.fi
git clone https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis.git \
  /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git clone <private remote> /scratch/project_2009497/LaclauGPT-Private

./LaclauGPT-Multimodal-Analysis/scripts/roihu/install_roihu.sh
```

What `install_roihu.sh` does:

1. fails fast (`set -euo pipefail`) and verifies it is in the right checkout;
2. loads `python-pytorch` and `gcc/14.3.0 ffmpeg` (Roihu exposes ffmpeg behind a
   GCC toolchain, so a bare `module load ffmpeg` is ambiguous);
3. creates a venv at `.venv-roihu-gpu/` (gitignored) and installs
   `requirements-roihu-steps1-6.txt`;
4. installs Ollama **for ARM64** under the private root if it is not already
   there, and puts it on `PATH`;
5. points `HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` at
   `/scratch/<project>/cache/…` so a job cannot fill `$HOME`;
6. validates that the real imports resolve — including `nemo.collections.asr`
   and `easyocr`, which is where a half-installed environment shows up;
7. prints the venv, private root and Ollama path.

Re-running is safe: an existing venv is reused.
`LACLAUGPT_ALLOW_NON_ROIHU_INSTALL=1` lets it run off-Roihu for testing.

## Private settings

**Nothing secret is committed to the public repository.** Credentials, Mongo
URIs, private dataset paths and country-specific settings live in the private
checkout, in a `.env` file:

```bash
# /scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/.env
export LACLAUGPT_MONGODB_URI=mongodb://...
LACLAUGPT_EP24_ROOT=/scratch/project_2009497/LaclauGPT-Private/ep24_reprocess
```

The path is resolved from `LACLAUGPT_EP24_ENV_FILE`, defaulting to `.env` under
the private root. `install_roihu.sh` sources it (`set -a`) so the values reach
the job environment; the pipeline steps load it via `ep24_settings.load_private_env()`,
which uses `setdefault` so an existing environment value always wins.

### Reporting settings without leaking them

```bash
python ep24_settings.py                  # uses the resolved private root
python ep24_settings.py --private-root /scratch/project_2009497/LaclauGPT-Private
python ep24_settings.py --env-file /path/to/.env
```

Output is **safe to paste into a ticket**: any setting whose name contains
`URI`, `URL`, `KEY`, `TOKEN`, `PASSWORD`, `SECRET`, `CREDENTIAL`, `CONNECTION`,
`DSN` or `PRIVATE` is shown as `<redacted N chars>`. Exit code is 1 when a
required setting is missing (`LACLAUGPT_MONGODB_URI`), so it can gate a job.

## Running the steps

Submit **one step at a time**, in order:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch
sbatch scripts/roihu/step_2_roihu_frame.sbatch
sbatch scripts/roihu/step_3_roihu_video.sbatch
sbatch scripts/roihu/step_4_roihu_summary.sbatch
sbatch scripts/roihu/step_5_roihu_postprocess.sbatch
sbatch scripts/roihu/step_6_roihu_discourse_analysis.sbatch
```

All six use `--partition=gpumedium`, `--gres=gpu:gh200:1`, `set -euo pipefail`,
and log to `/scratch/project_2009497/logs/ep24_s<N>_%j.{out,err}`.

Each script supports a country and a row limit, so a smoke test is one flag away:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch --country finland --limit 10
```

### A limited run is not a checkpoint

`--limit` writes to a distinct sample output path, so a truncated dataframe
cannot be mistaken for a complete-country result. The limit is recorded in the
run's provenance. Do not compare a sample run's row count with a full one.

### What the log tells you

Every step sources `scripts/roihu/lib_roihu_diagnostics.sh` and prints, before
doing any work:

```text
hostname / date / slurm_job_id / slurm_partition / slurm_cpus
public_root / private_root
git_commit (+ a "git_dirty" warning when the tree has uncommitted changes)
python + path, hf_home, tmpdir, ffmpeg version
gpu name + memory, driver version
package versions: pandas, torch, numpy, cv2, pymongo, redis
```

That is enough to reconstruct a failed run without re-running it — in particular
the git SHA, because a run from a dirty tree is not reproducible from a commit.

## The boundaries

The row is cumulative: every step takes the previous step's full row and appends
its own columns. Verify each producer's output is consumable by the next step
before moving on.

| step | script | reads | appends |
|---|---|---|---|
| 1 | `step_1_roihu_preprocess.py` | `data/to_reprocess/ep24_<country>.csv` (private, LFS) | `frame_file`, `frame_timestamp_seconds`, `ocr_1`, `ocr_*` metadata, `asr_*`, `video_duration_seconds`, `preprocess_*` |
| 2 | `step_2_roihu_frame.py` | step 1 output | `frame_analysis_1` |
| 3 | `step_3_roihu_video.py` | step 2 output + staged media | whole-video analysis fields |
| 4 | `step_4_roihu_summary.py` | step 3 output | `summary_analysis` + `laclau_*` summary fields |
| 5 | `step_5_roihu_postprocess.py` | step 4 output | `entities`, `topics`, sentiments, `formula_of_populism_*` |
| 6 | `step_6_roihu_discourse_analysis.py` | step 5 output | `laclau_structured_json`, `formula_of_populism_us` / `_frontier`, `laclau_*` |

Step 1 additionally **requires** the canonical `entities` and `themes` columns
and refuses input carrying the four legacy annotation columns, so an unmigrated
private CSV fails loudly instead of silently losing the annotation.

## Smoke test order

Run Finland → Poland → Portugal first, with a small `--limit`, and check the
boundary table above for each. Only then run the remaining countries and full
row counts.

## Troubleshooting

| symptom | cause |
|---|---|
| `Expected Roihu-GPU ARM64 login node` | you are on `roihu-cpu` or a local box |
| `Missing private EP24 root` | `LACLAUGPT_PRIVATE_ROOT` / `LACLAUGPT_MULTIMODAL_PRIVATE_ROOT` wrong |
| `WARNING: … required setting(s) missing` | `.env` not found or `LACLAUGPT_MONGODB_URI` unset |
| import error that makes no sense | venv built on the other architecture — rebuild on `roihu-gpu` |
| `Required canonical columns missing` | the private CSV has not been migrated (private #21) |
| `Unmigrated legacy annotation columns` | the private CSV still carries `new_entity` etc. |
| ffmpeg not found / module load fails | `gcc/14.3.0` must be loaded before `ffmpeg` on Roihu |
| `nemo.collections.asr` missing | `requirements-roihu-steps1-6.txt` did not fully install; re-run the script |
| job dies immediately | read `logs/ep24_s<N>_<jobid>.err`; the diagnostics block is the first thing in the log |
