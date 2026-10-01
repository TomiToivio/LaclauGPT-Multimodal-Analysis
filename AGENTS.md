# Agent rules

## Repository role

This repository has two deliberately different lines of history:

- `legacy`: immutable historical record of the original human-coded EP24 / CSC Puhti pipeline. Papers and publications may depend on this exact implementation.
- `main`: active **new EP24 analysis** for CSC Roihu using the **Phase 2 LaclauGPT pipeline**.

The old phase-branch synchronization policy does not govern this repository. Work for the current repository belongs on `main` unless the human author explicitly says otherwise.

## Legacy branch is immutable and must remain visible

Never modify, merge into, rebase, force-push, clean up, rename, reformat, modernize, or backport changes to `legacy`.

Do not delete or hide the legacy branch. Documentation on `main` must point readers to it because published research may rely on the historical implementation.

## Full legacy compatibility is mandatory

The human-coded legacy pipeline is the compatibility spine of the new implementation.

New Phase 2 features from `TomiToivio/LaclauGPT-Data-Analysis` must be added **around, before, after, or alongside** the legacy steps. Agents may extend legacy stages when necessary, but must not remove, replace, collapse, silently rewrite, or make a legacy step unavailable without explicit human permission.

Preserve, unless explicitly authorized otherwise:

1. the legacy-compatible logical sequence and the ability to run each historical stage inside the numbered Roihu pipeline;
2. existing input/output contracts and historical field meanings;
3. legacy-compatible CSV and cache outputs needed by existing research;
4. human-written prompts and Laclau/Palonen logic;
5. reproducibility links to the frozen `legacy` branch.

When modernization requires a changed implementation, retain a compatibility path or adapter.

## Human code is authoritative

Never replace human-written code merely because a rewrite appears cleaner. Make narrow, reviewable changes and validate them.

Do not perform opportunistic destructive refactors, schema removals, prompt rewrites, or methodology changes. When uncertain, preserve the human implementation and add the new functionality beside it.

## Phase 2 direction

`main` should progressively incorporate relevant features from [LaclauGPT-Data-Analysis](https://github.com/TomiToivio/LaclauGPT-Data-Analysis), including where technically and scientifically appropriate:

- newer/better local multimodal models supported on CSC Roihu;
- richer provenance and resumability;
- RDF graph outputs;
- Discourse Network Analysis (DNA);
- Social Network Analysis (SNA);
- improved validation and structured outputs;
- additional Phase 2 modules that can coexist with legacy-compatible EP24 stages.

Do not copy the newer pipeline blindly. Integrate it incrementally around the compatibility spine.

## Roihu naming

Active `main` code and documentation use **Roihu**, not Puhti, for current execution. Canonical active batch entry points are `step_N_roihu_*.py`; historical `roihu_*.py` filenames remain compatibility implementations.

Use “Puhti” only when describing the historical `legacy` environment, provenance, or compatibility history.

## Numbered Roihu batch contract

The active main-branch pipeline is numbered and each stage is a separate Slurm job:

1. preprocess
2. frame analysis
3. optional whole-video VLM
4. summary/fusion
5. postprocess
6. Laclau/Palonen discourse analysis
7. Discourse Network Analysis
8. Social Network Analysis
9. RDF export

Reserve Step 3 even while whole-video vLLM is experimental so downstream stage numbers remain stable. Do not renumber later stages when the video backend graduates.

For demos, honor `LACLAUGPT_MAX_ROWS=100`. A full run may override the limit. Each step must remain executable with a simple command such as `python3 step_7_roihu_discourse_network_analysis.py`, with a matching sbatch file under `scripts/roihu/`.

RDF is a deterministic post-analytic export and should normally use a CPU partition rather than consume a GH200 GPU.

## Public/private boundary

Public open-source code belongs here.

Sensitive or private material belongs in `TomiToivio/LaclauGPT-Private` and/or private CSC storage, including:

- real research data;
- researcher notes;
- codebooks;
- settings and private configuration;
- restricted prompts or unpublished mappings;
- credentials, secrets, and tokens;
- machine-specific sensitive configuration.

Never copy private content into this public repository to make a test or job work. Public code may define interfaces, schemas, environment variables, safe examples, and dummy fixtures.

## Development principle

**Preserve the published legacy record. Keep full compatibility. Build Phase 2 around it. Run the active pipeline on Roihu. Keep private research material private.**


## EP2024 storage and dataframe contract

For active EP2024 reprocessing on Roihu:

- MongoDB is the canonical shared durable research backend for dataframe mirrors/results, codebooks, memory, RAG, researcher notes, embeddings, graph/RDF/DNA/SNA material, provenance and backups.
- Redis is for transient coordination/cache/messaging/locks/status, not the durable source of truth.
- PostgreSQL is not part of this project architecture.
- SQLite and DuckDB are allowed only as local/job-local helpers, compatibility artifacts, imports/exports or checkpoints.
- Source videos remain in CSC Allas and are downloaded on demand from the URL/object identifier after the user configures `allas_conf`.
- Input and output remain Pandas-compatible CSV files. Preserve every legacy column and meaning.
- New analytical fields are additive. Every major step must provide a human-readable Markdown summary field as well as any structured JSON/machine output.
- Use MongoDB collection names `laclaugpt_ep2024_reprocess_<country_name>_<collection_name>`.
- Real MongoDB/Redis credentials are read from the private companion repository/runtime environment and must never be copied into this public repository or logs.


## EP24 video invariant

For EP24 TikTok/Instagram screen recordings, every split clip contains a known
scroll transition from the previous item in its first 1.0 second. All new media
analysis must import the canonical rule from ep24_video.py and exclude that
interval from ASR, frames, OCR, whole-video VLM input, embeddings, scene
sampling, summaries, fusion, and future video modules. Do not add independent
magic numbers.

Additional feed scrolls after the known artifact are data-quality failures.
Whole-video VLM output must preserve human-readable analysis and expose SCROLL
and SCROLL_SECONDS. Detected additional scrolls enter the deterministic
needs_resplit workflow with source provenance and legacy fields preserved.
Never overwrite source media, never modify the legacy branch for this rule, and
keep recursion bounded.
