# LaclauGPT Coding Style

> **Status:** documentation of the code style that `LaclauGPT-Multimodal-Analysis`
> already follows. Nothing here changes pipeline behaviour, and nothing here is a
> rule the code does not already demonstrate.
>
> **Guiding principle: optimize for the researcher reading the code six months
> later.** Readable research software beats clever research software.

This is a research pipeline, not a product. The code must stay understandable to
researchers and future maintainers, so readability and research transparency take
priority over abstraction.

The same principles are documented in `LaclauGPT-Data-Analysis`, because
Multimodal-Analysis is intended to be the architectural ancestor of the rebuilt
AI26 Data-Analysis pipeline. Each repository's copy states that repository's
*actual* current conventions; the two are at different stages (see
[Repository status](#repository-status)).

## 1. One analysis step = one clearly named file

Each major analysis step lives in its own Python file, directly executable from the
command line, and readable in one sitting.

This repository already works this way:

```text
step_1_roihu_preprocess.py
step_2_roihu_frame.py
step_3_roihu_video.py
step_4_roihu_summary.py
step_5_roihu_postprocess.py
step_6_roihu_discourse_analysis.py
step_7_roihu_discourse_network_analysis.py
step_8_roihu_social_network_analysis.py
step_9_roihu_rdf.py
```

The `step_<N>_roihu_<name>.py` shape is deliberate: the number gives the pipeline
position, `roihu` names the platform batch, and the suffix names the operation. A
step must remain runnable as:

```bash
python step_4_roihu_summary.py
```

CLI options, sbatch and cron wrappers are additive. They must never become the
*only* way to run a step.

**The stage contract must stay truthful.** `ep24_stage_contract.py` names each
step's executable entry point and the dataframe columns it appends. If a step
gains a column or changes its entry point, the contract is updated in the same
change — a contract that under-declares a stage is worse than no contract, because
other code trusts it.

## 2. Shared infrastructure lives in separate, clearly named modules

Code genuinely shared by several steps is factored into named modules rather than
duplicated. In this repository that is the `ep24_*.py` family:

```text
ep24_pipeline.py     cumulative CSV contract, metadata context, additive writes
ep24_schema.py       canonical schema, stable id derivation
ep24_entities.py     mention normalisation and entity resolution
ep24_memory.py       persistent memory
ep24_db.py           MongoDB access
ep24_redis.py        Redis access
ep24_cli.py          shared step CLI (country selection, input/output paths)
ep24_settings.py     runtime configuration
ep24_models.py       model defaults and routing
```

Rule of thumb:

> **Shared infrastructure should be abstracted. Research logic should remain
> visible.**

The step file stays a readable entry point. Do not move the actual logic of a step
into a maze of helper classes merely to reduce its length.

## 3. Prefer readability over architectural cleverness

Prefer:

- straightforward functions
- explicit control flow
- descriptive variable names
- obvious inputs and outputs
- simple modules
- visible prompts
- visible pipeline stages

Avoid, unless the author asks for it:

- class hierarchies
- factories
- plugin frameworks
- dependency-injection machinery
- deeply nested utility layers
- premature abstraction
- "enterprise" architecture for its own sake

```python
# yes — the reader sees what happens and in what order
rows = load_cumulative_csv(path)
for index, row in rows.iterrows():
    analysis = analyse(row)
    frame.at[index, "summary_analysis"] = analysis
write_cumulative_csv(original, frame, output)

# no — the same work, hidden behind indirection the reader must chase
write_cumulative_csv(original, StepPipeline(StepContext(row)).run(), output)
```

## 4. Heavy debug logging is a feature

Extensive info/debug logging is intentional. A human must be able to answer:

> "What exactly is this step doing right now, with what data, and where is the
> result going?"

Log step start/end, input files and database collections, country/sample/filters,
record counts, the current record where useful, model used, prompt stage, fields
passed between steps, output destinations, database and backup writes, retries and
fallbacks, validation failures, and exceptions with context. Use the shared
`LACLAUGPT_LOGLEVEL` convention rather than a bespoke logging setup.

## 5. Use generous comments and docstrings

Every step should carry a module docstring covering: purpose, pipeline position,
expected inputs, outputs produced, databases/files used, models used, important
assumptions, and CLI examples. Comment non-obvious transformations, and comment
research and theory decisions.

Comments explain **why**, not only **what**.

## 6. System prompts may be long and theory-heavy

Heavy system prompts are intentional and are research code. LaclauGPT is not only
an ETL pipeline: its prompts encode theoretical and methodological assumptions
(Laclau, Castells, Leifeld/DNA, SNA, ideology, discourse theory, populism,
assemblages). Therefore:

- keep prompts readable in the source
- do not aggressively shorten them
- do not move them into opaque abstractions for code neatness
- preserve theoretical context
- keep research concepts visible to reviewers

Prompt readability is part of code readability.

## 7. Keep data flow explicit

Every step must make it obvious what it reads, which fields it expects, which new
fields it creates, what it preserves, what it writes, and what the next step
receives.

**New analytical fields are additive.** A step preserves all previous fields and
appends its own outputs, via the shared cumulative write helpers rather than a
direct `to_csv`. Every major step also provides a human-readable Markdown summary
field alongside any structured JSON, so a researcher can read the result without
parsing it.

Avoid hidden mutations and implicit schema changes.

## 8. Self-contained does not mean duplicated infrastructure

A step must be understandable as a standalone research operation while still being
allowed to import shared infrastructure:

```python
from ep24_pipeline import load_cumulative_csv, write_cumulative_csv
from ep24_entities import fold_key, resolution_lookup
from ep24_models import ollama_model
```

This keeps the research step visible while sharing boring infrastructure. What to
avoid is importing the pipeline's plumbing instead of writing the step:

```python
from framework.runtime.pipeline.step_factory import AbstractStepProvider
```

## 9. Tests and verification

Wherever practical, a behavioural change is demonstrated by **executing** the code,
not by reading it. For a guard you add, prove it fails when the defect is
reintroduced, then restore the file and confirm it passes again. A guard whose
sabotage still passes is a broken guard, not reassurance.

## 10. Migration direction (EP24 → AI26)

Once the Roihu EP24 reprocessing pipeline works end-to-end:

1. preserve the working EP24 implementation and its history;
2. use the Multimodal-Analysis structure as the basis for modernizing
   `LaclauGPT-Data-Analysis`;
3. adapt that code from EP24 to AI26 rather than reverting to the older
   Data-Analysis architecture;
4. preserve this simple, step-oriented style throughout the migration.

> **Multimodal-Analysis becomes the architectural ancestor of the rebuilt AI26
> Data-Analysis pipeline.**

**Status: not started.** No EP24→AI26 port has been performed, and the Roihu
pipeline is not yet end-to-end complete. This section records the direction, not a
completed migration.

## Repository status

The two repositories are at different stages, and the guide says so rather than
pretending otherwise:

| | `LaclauGPT-Multimodal-Analysis` | `LaclauGPT-Data-Analysis` |
|---|---|---|
| steps | `step_1_roihu_*.py` … `step_9_roihu_rdf.py`, implemented | `laclaugpt/step_01_*.py` … `step_07_sna.py`, **Tomi-locked and pending hand-coding** |
| infrastructure | `ep24_*.py` | `laclaugpt/laclaugpt_*.py`, `pipeline_*.py` |
| theory prompts | in the step modules | `laclaugpt/PROMPT_LACLAU.md`, `PROMPT_DNA.md`, `PROMPT_SNA.md` |
| legacy reference | `puhti_*.py` | `laclaugpt/puhti_*.py` |

### What this guide does not override

Repository-specific agent rules win where they are narrower. In particular:

- **`LaclauGPT-Data-Analysis`** marks `laclaugpt/step_01…step_07` **TOMI-LOCKED**
  (see its `AGENTS.md` and `laclaugpt/HUMAN_PIPELINE.md`). Agents work *around*
  those files — I/O, contracts, prompts, RDF/export, orchestration, tests,
  deployment, adapters — and must not modify, rename, merge, replace or move them
  without explicit authorisation for that specific step.
- **`LaclauGPT-Data-Analysis`** also prioritises machine-readable canonical records
  over human-readable **reports**. That is a decision about pipeline *output*;
  this guide is about code *readability*. The two are compatible, and nothing here
  reopens the output question.

## See also

- [EP24 pipeline contract](EP24_PIPELINE_CONTRACT.md)
- [Roihu numbered pipeline](ROIHU_NUMBERED_PIPELINE.md)
- [Agent rules](../AGENTS.md)
