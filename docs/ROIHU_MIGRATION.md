# CSC Roihu migration

## Branch contract

- `legacy` is the frozen historical CSC Puhti / EP24 implementation.
- `main` is the active CSC Roihu implementation.

At the start of issue #4, `legacy` and `main` both pointed to commit
`e011c34274c41923e801c32c824c5afa7468e1a1`. The migration starts by diverging `main`; `legacy` must not move.

## Stage 0 inventory

The historical pipeline is five sequential Python scripts:

1. `puhti_preprocess.py`
2. `puhti_frame.py`
3. `puhti_summary.py`
4. `puhti_postprocess.py`
5. `puhti_populism.py`

The existing code uses relative runtime paths such as `./csv`, `./Allas`, `./Keyframes`, `./database`, `./logs`, and `./whisper`. The Roihu baseline deliberately preserves those human-written paths by executing the public scripts with the **private runtime directory as the current working directory**.

Historical LLM model names are hard-coded in the inference stages. The Roihu baseline changes only model selection, via `LACLAUGPT_MULTIMODAL_MODEL`, while preserving prompts, output fields, stage order and analytical logic.

## Public/private contract

Public repository: `TomiToivio/LaclauGPT-Multimodal-Analysis`

Private repository: `TomiToivio/LaclauGPT-Private`

Set:

```bash
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT=/path/to/LaclauGPT-Private/analysis/ep24-multimodal
```

The private runtime root is expected to contain `csv/`, `Allas/`, `Keyframes/`, `database/`, `logs/`, `whisper/`, and `.ollama/`.

Private codebooks, settings, researcher notes and source data remain in `LaclauGPT-Private`. They are not copied into this public repository.

## Roihu platform assumptions

The public batch template targets one full GH200 GPU on `gpumedium`. Roihu GPU nodes use ARM/aarch64 CPUs, so a Puhti x86 virtual environment must not be copied to Roihu.

The batch script loads CSC's `python-pytorch` module and `ffmpeg`, then expects a Roihu-created virtual environment. Site allocation IDs and absolute private paths are intentionally not committed.

Create the virtual environment on `roihu-gpu.csc.fi`, for example:

```bash
module --force purge
module load python-pytorch
python -m venv --system-site-packages .venv-roihu-gpu
source .venv-roihu-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Gemma4 baseline

Initial configurable default:

```bash
export LACLAUGPT_MULTIMODAL_MODEL=gemma4:12b
```

This is a baseline candidate, not a scientific-method change or permanent model lock. No cloud fallback is used.

## Submit

Keep the CSC project allocation outside tracked scripts:

```bash
export CSC_ACCOUNT='<project>'
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT='/private/path'
sbatch --account="$CSC_ACCOUNT" scripts/roihu/multimodal_roihu.sbatch
```

For the first smoke run set `LACLAUGPT_MULTIMODAL_STAGES=frame`. Supported stages are `preprocess frame summary postprocess populism`.

## Current validation status

Public CI can validate syntax, model configurability, the batch template and the legacy-branch immutability contract. It cannot execute a GH200/Ollama research run because GitHub Actions has neither CSC Roihu nor private EP24 data.

Therefore a real Gemma4 test remains a Roihu execution step. Record the tested model, Slurm job ID/resource request, stage, result and output location in the issue or a private run note.

## Output compatibility

The baseline intentionally preserves the five-stage order, existing prompts, existing CSV field names, existing SQLite cache behavior, and legacy `ep24_<language>.csv` compatibility output from postprocess.

## Candidate improvements from LaclauGPT-Data-Analysis

Candidates only, not automatically ported:

1. job-local Ollama startup and fail-fast model checks;
2. ARM64-aware Roihu environment setup;
3. explicit public/private root contracts;
4. stage-level smoke modes and provenance logging;
5. deterministic ffmpeg keyframe extraction;
6. stronger cache fingerprints and resumable provenance;
7. local-only ASR alternatives such as faster-whisper;
8. explicit no-cloud-fallback LLM routing;
9. richer validation of actual image evidence.

For each candidate, preserve the working Roihu baseline first, compare behavior explicitly, and implement only through a later narrow change.
