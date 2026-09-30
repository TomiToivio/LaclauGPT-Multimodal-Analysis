# Puhti → Roihu migration: Stage 0 inventory

This is the Stage 0 deliverable for issue #4: inspect and document **before**
changing any code.

Status: inventory and plan only. No pipeline behaviour is changed by this document.

---

## 1. Repository state

| | |
|---|---|
| Default branch | `main` |
| `main` HEAD at inspection | `e011c34` — `docs: distinguish EU and Research Council of Finland funding in logo layout` |
| Tracked files | 15 |
| Pipeline modules | 5 (`puhti_*.py`, 1454 lines total) |

### Finding 1: `legacy` is not yet a frozen historical branch

The issue describes `legacy` as *"immutable historical documentation of the original
EP24 / CSC Puhti multimodal pipeline"* and `main` as *"careful, incremental
modernization"*. In the repository right now:

```
$ git rev-list --left-right --count origin/legacy...origin/main
0        0
```

`legacy` and `main` point at **the same commit**. There is no divergence, so there is
nothing frozen that `main` has moved away from.

This matters because the issue's central safety rule — *"Do not modify, rebase, merge
into, force-push, rewrite, or otherwise alter the `legacy` branch"* — currently
protects a ref that is indistinguishable from the branch we are about to change. If
work proceeds on `main` without first pinning `legacy`, then `legacy` will silently
stop being "the historical implementation" the moment the first `main` commit lands.

**Recommendation (needs your decision, since it is a branch operation):** before any
`main` change, record the frozen ref explicitly — either leave `legacy` where it is
and treat `e011c34` as the frozen content (it already is byte-identical), or tag it
(e.g. `legacy-puhti-e011c34`) so the provenance is unambiguous and independent of the
branch pointer. I have not done either, because branch/tag creation is your call and
the issue forbids touching `legacy`.

### Other branches present

| Branch | vs `main` |
|---|---|
| `fix/legacy-pipeline-bugs` | 3 commits ahead, 22 behind |
| `python-standards` | 0 ahead, 9 behind |
| `phase-0` | 0 ahead, 3 behind |
| `phase-1` … `phase-4` | not inspected in detail |

`fix/legacy-pipeline-bugs` holds **3 commits not on `main`**. Worth a look before
anything is rewritten — one of them may be a genuine bug fix that should be carried
forward rather than lost. Not in scope for this issue as written.

---

## 2. Branch-policy conflict (must be resolved explicitly)

`AGENTS.md` currently declares **Phase 0 is active** and that `main` must stay
synchronised with `phase-0`. `docs/PHASE_BRANCHING.md` repeats it.

Issue #4 says:

> Do **not** follow any older repository instruction that says `main` must mirror
> Phase 0 or another phase branch when that conflicts with this issue.

So the issue supersedes the policy *for this migration*. That is a real conflict
between two authoritative documents, and the resolution should be recorded in the
repository rather than left implicit — otherwise the next agent reading `AGENTS.md`
will do the wrong thing. I have **not** edited `AGENTS.md` (it is an agent-instruction
file requiring your approval); §7 proposes the change.

---

## 3. Puhti-specific assumptions in the current pipeline

Everything below is what a Roihu port has to deal with. Counted from the five modules.

### 3.1 Paths — all relative to the working directory

| Path | Used by | Purpose |
|---|---|---|
| `./logs/` | all five | rotating log files, created at import time |
| `./database/preprocess.db`, `frame.db`, `summary.db` | 1, 2, 3 | per-stage SQLite state |
| `formula_of_populism.db` | 5 | **not** under `./database/` — inconsistent |
| `./csv/tiktok_videos.csv`, `./csv/tiktok_{language}.csv` | 1, 2, 3, 4 | stage hand-off via CSV |
| `./Keyframes/TikTok/{user}/{video_id}/` | 1 | extracted frames |
| `./whisper/` | 1 | Whisper model download root |
| `./Allas/Scraper/TikTok/Videos/{country}/` | 1 | source media staging |

Two problems for Roihu:

- **`./logs` and `./database` are created at import time** (`os.makedirs(...)` at module
  top level). On a cluster that means importing a module writes to the CWD, which is
  wrong for a batch job and makes the modules untestable without side effects.
- Relative paths assume the job's CWD. Roihu batch jobs should resolve paths from
  configuration, not from where `sbatch` happened to be run.
- `formula_of_populism.db` sits beside the scripts while every other DB is under
  `./database/` — noted as an inconsistency to preserve deliberately rather than
  "fix" incidentally.

### 3.2 Hard-coded model identifiers

Two distinct identifiers at four call sites:

```
llama3.2-vision:11b   -> puhti_frame.py:110, puhti_summary.py:170
gemma3:27b            -> puhti_postprocess.py:68, puhti_populism.py:303
```

These are literals inside `ollama.chat(...)` calls. Stage 2 requires a configurable
Gemma4 target, so these must become configuration — that is the *only* change Stage 2
sanctions to those lines. Note `puhti_populism.py:302` carries a comment listing
alternatives (`llama3.3:70b`, `qwen3:32b`, …), which is evidence that model choice was
already treated as adjustable rather than fixed.

### 3.3 Ollama usage

All four LLM stages call `ollama.chat(...)` directly through the `ollama` Python
package. There is no endpoint configuration, and therefore no way to point the
pipeline at a Roihu-local Ollama server without editing code. No `OLLAMA_HOST`
handling exists.

### 3.4 Compute dependencies

| Dependency | Stage | Note |
|---|---|---|
| `opencv` (`cv2`) | 1, 2 | frame extraction |
| `easyocr` | 1 | OCR on frames |
| `openai-whisper` | 1 | audio transcription; `whisper.load_model('large')` downloads to `./whisper/` |
| `deep_translator` | 1 | Google translation of non-English transcripts |
| `pandas` | 1, 2, 3, 4, 5 | CSV hand-off |
| `sqlite3` | 1, 2, 3, 5 | stdlib; per-stage state |
| `ollama` | 1, 2, 3, 4, 5 | LLM calls |

`ffmpeg` is required by Whisper and OpenCV. The Roihu module stack already known to
this organisation is `gcc/14.3.0`, `python-pytorch/2.13`, `ffmpeg` (from the sibling
`LaclauGPT-Data-Analysis` Roihu runbook) — that is a **starting point to verify**, not
an assumption to bake in.

### 3.5 Batch submission

**There is no sbatch script in this repository.** The README describes the pipeline as
"submitted as batch jobs in a sequence" and links to CSC's Puhti batch-job
documentation, but no job script is tracked. So the SLURM layer is not being ported —
it is being **written for the first time**. That is the largest single piece of
missing infrastructure and it belongs to Stage 1.

### 3.6 Sequential `for language in languages` / `for country in countries`

Each stage loops over hard-coded country/language lists at module level, e.g.
`for language in languages:` at the bottom of `puhti_preprocess.py`. Consequence: a
"job" is one Python process doing every country in sequence. On Puhti that presumably
fitted the allocation; on Roihu it determines whether one job or an array job is the
right shape. Flagged as a Stage 1 decision, not changed here.

---

## 4. Private dependencies that must come from `LaclauGPT-Private`

The pipeline reads real research material. From the code and the sibling repositories:

| Material | Where it belongs |
|---|---|
| Source media (`Allas/Scraper/TikTok/Videos/...`) | private storage / CSC project area |
| Extracted frames (`Keyframes/`) | runtime only, never committed |
| CSV hand-off files (`csv/tiktok_*.csv`) | runtime only — contain real rows |
| Per-stage SQLite DBs | runtime only — contain real rows |
| EP24 codebooks, actor/entity mappings | `LaclauGPT-Private` |
| EP24 prompts containing research-sensitive material | `LaclauGPT-Private` |
| Any credentials / `OLLAMA_HOST` / project account | private config, never public |

`.gitignore` already excludes `Keyframes`, `Allas`, `database`, `logs`, `whisper` from
linting, and those directories are untracked. Confirmed no real data is tracked on
`main` today.

**Public** may contain: the pipeline code, schemas, dummy/example data, example config,
documented expected paths and environment variables, tests, and deployment templates
without secrets.

---

## 5. Roihu environment — what is known, and what must be verified

### 5.1 The sibling repository already runs EP24 on Roihu

Stage 0 asks for `LaclauGPT-Data-Analysis` to be inspected "only for later
compatibility/reference". That inspection produced the single most useful finding in
this document, because **the same study (EP24 Finland/Poland) already has a working
Roihu implementation there**:

| Artefact | What it provides |
|---|---|
| `src/laclaugpt_data_analysis/ep24_roihu.py` | study-specific Roihu runner for EP24 multimodal Phase 2; `DEFAULT_MODEL = "gemma4:12b"` |
| `scripts/ep24/ep24_roihu_reprocess.sbatch` | **complete EP24 Roihu batch script** — partitions, modules, ARM64 Ollama, private-root checks |
| `requirements/roihu-ep24.txt` | dependency set already used for EP24 on Roihu |
| `src/laclaugpt_data_analysis/hungary26_roihu.py` | reusable primitives: `ollama_chat()` (honours `OLLAMA_HOST`), `extract_keyframes()`, `sha256_file()`, `private_paths()`, `ensure_private_layout()` |
| `scripts/roihu/reprocess_project.sbatch` | generic parameterised template with no project values baked in |

The EP24 script resolves these known-answer questions that I would otherwise have had
to send you:

```bash
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00

module --force purge
module load gcc/14.3.0
module load python-pytorch/2.13
module load ffmpeg
```

and the Ollama pattern (per-job port, private model cache, background serve with
trap-based cleanup and a readiness loop):

```bash
OLLAMA_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OLLAMA_MODELS=${OLLAMA_MODELS:-${PRIVATE_ROOT}/.ollama/models}
export OLLAMA_HOST="http://127.0.0.1:${OLLAMA_PORT}"
export LLM_MODE=local
export LLM_ALLOW_CLOUD_FALLBACK=0
ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT INT TERM
for _ in {1..60}; do ollama list >/dev/null 2>&1 && break; sleep 2; done
```

Also note the ARM64 reality is handled head-on there: the script refuses to install an
Ollama build unless `uname -m` is `aarch64`/`arm64`, and pulls the ARM64 tarball. That
matches CSC's statement that Roihu is ARM64.

**Consequence for Stage 1:** the sbatch layer is no longer unwritten work to design
from scratch — it is a template to adapt, and the adaptation is mostly a matter of
mapping the five `puhti_*.py` stages onto this proven pattern rather than inventing
cluster configuration. This should be treated as **deployment infrastructure to reuse**,
not copied analysis code, and the issue is explicit that the whole Data-Analysis
pipeline must not be copied over.

**One difference to reconcile deliberately:** that pipeline uses `faster-whisper`,
while this repository's `puhti_preprocess.py` uses `openai-whisper` with an explicit
`./whisper/` model download root. Whether to keep `openai-whisper` (no behaviour
change, larger ARM64 dependency risk) or switch to the already-proven `faster-whisper`
is a Stage 1 decision that changes a dependency, not scientific logic. It is listed as
an open question in §8 rather than decided here.

### 5.2 Known from this organisation's existing Roihu work

From `LaclauGPT-Data-Analysis` (`docs/RESTRICTED_ROIHU_REPROCESSING.md`,
`scripts/ep24/ep24_roihu_reprocess.sbatch`), which already runs on Roihu:

```bash
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-12:00:00

module load gcc/14.3.0
module load python-pytorch/2.13
module load ffmpeg
```

Also noted there: Roihu GPU jobs use an **ARM64** environment built on Roihu itself.

### From CSC's own documentation

Puhti and Mahti are **decommissioned**; their login/storage nodes remain only until
15 October 2026, with migration to Roihu explicitly advised. So this migration is not
optional housekeeping — the current pipeline's home is being switched off.

Roihu documentation covers: partitions, batch-job creation, example scripts,
Apptainer/Tykky containers, Roihu disk areas (parallel filesystem, local NVMe), and
model/dataset storage. The specifics still to be pinned down for this pipeline are in
§6.

### Must be verified on Roihu before Stage 1 is called done

- exact partition(s) and GPU types available, and what `--gres` string each expects;
- whether the pipeline's CUDA/torch expectations match Roihu's ARM64 environment —
  `easyocr` and `openai-whisper` both pull heavy native deps and are the likeliest
  portability blockers;
- how Ollama is to run on a compute node (no daemon assumption can be inherited from
  Puhti);
- where models live (parallel filesystem vs local NVMe) and how Ollama's model cache
  is pointed at it;
- per-job scratch and the correct way to stage media in and results out;
- whether `deep_translator` (Google) is reachable from Roihu compute nodes — if not,
  translation becomes an explicit optional step.

---

## 6. Stage plan

Following the issue's required order. Each stage is a separate, reviewable step.

### Stage 1 — Roihu baseline (smallest possible infrastructure change)

1. Add a configuration surface for paths (input root, output root, scratch, model cache,
   Ollama endpoint, per-stage model ids). Environment variables with documented
   defaults; no personal paths in the public repo.
2. Add sbatch templates — one per stage, or one parameterised script — **adapted from
   the proven `scripts/ep24/ep24_roihu_reprocess.sbatch` pattern in
   `LaclauGPT-Data-Analysis`** rather than designed from scratch (§5.1). The directives
   there are known-good for this organisation's Roihu allocation.
3. Move the import-time `os.makedirs` calls behind the config so importing a module has
   no filesystem side effects.
4. Keep every stage's scientific logic byte-identical.

### Stage 2 — Gemma4

Replace the four hard-coded model literals with configuration and run a smoke test.
Report: model identifier actually available, resource request used, Ollama
configuration, known limitations, which stages use the multimodal model. **Do not
change prompts or analysis to suit the model.**

### Stage 3 — validate before improving

Per-stage validation, then an end-to-end run on dummy or approved private input.
Confirm output structure matches the historical EP24 workflow, and document every
unavoidable difference.

### Stage 4 — evaluate `LaclauGPT-Data-Analysis` improvements

Written candidate list first, one row per candidate with current behaviour, newer
behaviour, why it helps *this* pipeline, and compatibility risk. Port only what is
justified, one at a time.

---

## 7. Proposed documentation changes (not yet made)

1. **`README.md`** — state the branch roles explicitly:
   `legacy` = original CSC Puhti EP24 pipeline, frozen for research documentation;
   `main` = active CSC Roihu version. Preserve the existing historical narrative and
   link to `legacy` rather than replacing it.
2. **New `docs/ROIHU_MIGRATION.md`** — operational detail (paths, modules, sbatch,
   model config), so the README does not accumulate it.
3. **`AGENTS.md` / `docs/PHASE_BRANCHING.md`** — resolve the branch-policy conflict
   §2 records. Both are agent-instruction files; editing them needs your approval and
   I have not touched either.

---

## 8. Open questions for the author

1. **`legacy` pinning** — tag the frozen commit, or rely on the branch pointer as-is?
   (Nothing is lost either way today; the risk is future divergence.)
2. **`fix/legacy-pipeline-bugs`** — its 3 commits are not on `main`. Carry them
   forward, or leave them? They are large (291 insertions in `puhti_preprocess.py`,
   147 in `puhti_postprocess.py`), and one is titled *"Preserve existing legacy Whisper
   results"* — which sounds like it prevents re-transcription work. Losing them would be
   a real cost, so this deserves an explicit answer rather than a default.
3. **Scope of Stage 1** — is one parameterised sbatch script preferred (mirroring
   `scripts/roihu/reprocess_project.sbatch`), or one per stage? This determines how the
   job sequencing in §3.6 is expressed.
4. **Ollama on Roihu** — the pattern in §5.1 already answers this technically (per-job
   port, `OLLAMA_MODELS` under the private root, ARM64 install, trap cleanup). The open
   part is whether this project should start its own server per job as EP24 does, or
   attach to a shared one.
5. **Private path contract** — confirm the expected `LaclauGPT-Private` layout for EP24
   codebooks and settings. The sibling repo already enforces a concrete contract
   (`source/`, `codebooks/`, and `ep24_common_private.json`,
   `ep24_finland_private.json`, `ep24_poland_private.json`), so the question is whether
   this repository should adopt that same layout or define its own.
6. **Whisper dependency** — keep `openai-whisper` (no behaviour change) or adopt the
   already-proven `faster-whisper` from `requirements/roihu-ep24.txt`? Both are
   dependency-level changes; the second is the one with evidence behind it on ARM64.
7. **Model identifier** — the sibling EP24 Roihu runner defaults to `gemma4:12b`. Stage 2
   says a specific Gemma4 size must not become irreversible; is `gemma4:12b` the
   intended first target here too, or should the smoke test sweep sizes?
