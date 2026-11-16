# EP24 refactor audit (#411): duplication, orphans and the order of work

**Status:** audit only. Nothing is moved, deleted or renamed here; no prompt, schema, CLI or output
changes. This is deliverable 1 of #411 ("before/after audit"), written so the rest of the plan can be
refused, narrowed or reordered before any code moves.

**Where the module-by-module map lives:** a separate deliverable under `docs/` covers the repository
map (which module is which, which step calls what). This document does not repeat it; it covers what
should happen to those modules, and what cannot be decided by an agent.

**Method:** every number comes from `main` at `858f383` — AST import graph, identical-function-body
hashing across the 69 top-level modules, reference search over every `*.py`/`*.sh`/`*.md`/CI file, and
line counts.

---

## Verdict, read this first

Three findings change the shape of the epic before its tasks are worth starting.

### 1. The layering is deliberate, and one layer may not be touched

| layer | modules | role | status |
| --- | --- | --- | --- |
| numbered entry points | `step_0_…`–`step_9_…` (10 files, 1.6 k lines; 84 % of that in steps 7–8) | one file per stage; the operator surface | canonical |
| active implementations and shared code | `roihu_*`, `ep24_*`, `asr_backend`, `ocr_backend`, `laclaugpt_quality` | what the numbered steps call | canonical |
| legacy stage implementations | `puhti_preprocess`, `puhti_frame`, `puhti_summary`, `puhti_postprocess`, `puhti_populism` (1.3 k lines) | human-coded EP24 stages, importable and tested from `main` | **explicit legacy boundary** |
| ad-hoc tooling | `diagnose_ep24_step6.py`, `run_step6_fixed.sh` | manual debugging | unclassified — see §3 |

`AGENTS.md` is unambiguous about the third layer: "must not remove, replace, collapse, silently
rewrite, or make a legacy step unavailable without explicit human permission". A plan that
"consolidates the parallel implementations" is, for `puhti_*`, a request to retire the compatibility
spine. That is a ruling, not a refactor — which is why the plan in §5 is split on that line.

### 2. Textual duplication is small; the epic over-estimates it

Across all 69 top-level modules, **four** function bodies are identical in two different modules
(§2). Everything else that looks duplicated is a *different implementation of a similar
responsibility*: the legacy/active pairs share 2–11 % of their lines.

```
puhti_preprocess.py   216 lines   roihu_preprocess.py   763 lines   line similarity 0.02
puhti_frame.py        207 lines   roihu_frame.py        495 lines   line similarity 0.11
puhti_summary.py      320 lines   roihu_summary.py      741 lines   line similarity 0.04
puhti_postprocess.py  191 lines   roihu_postprocess.py  623 lines   line similarity 0.04
puhti_populism.py     398 lines   roihu_populism.py     618 lines   line similarity 0.02
```

And one pair the epic would naturally merge is documented as intentionally distinct:

> "Their identity rules and manifests differ; do not combine their record counts or substitute one
> export directory for the other without validating the consuming application."
> — `docs/EP24_MONGO_EXPORT.md:77`, on `ep24_mongo_export.py` vs `ep24_mongo_dashboard_export.py`

So the true surface of "duplicate and obsolete implementations" is **three orphans and one genuine
copy cluster** (§2, §3) — not a repository-wide collapse.

### 3. Documentation and inventory can start now; removals cannot

`AGENTS.md` and `docs/CODING_STYLE.md` forbid opportunistic destructive refactors, schema removals,
prompt rewrites and "architectural cleverness", and make human-written code authoritative. That does
not contradict #411 — it fixes the order: measure and document first, remove and consolidate only on
an explicit ruling. §5 is split exactly there.

---

## 1. Import structure, and the one structural inconsistency

`ep24_schema` and `ep24_cli` are each imported by 12 modules, `roihu_storage` by 10,
`ep24_pipeline` and `ep24_logging` by 9. There is one orchestration path
(`ep24_stage_orchestrator` + `ep24_stage_state`), not two — nothing in the graph suggests a second
pipeline running in parallel.

The step-per-file convention, however, holds for **eight of the ten entry points**. Steps 0–6 and 9
are 10–137-line orchestrators that parse arguments and call an implementation module. Steps 7 and 8
implement in place:

```
step_0_roihu_import.py                        137
step_1 … step_6 (6 files, 10–39 lines each)   106
step_7_roihu_discourse_network_analysis.py    857   ← stage implementation in the entry point
step_8_roihu_social_network_analysis.py       517   ← stage implementation in the entry point
step_9_roihu_rdf.py                            17
                                        total 1,634   (84 % of it in steps 7 and 8)
```

That is the single clearest structural inconsistency in the repository, and it is also where the
duplicated helpers live (§2). Fixing it means extracting step 7's and step 8's bodies into
`roihu_discourse_network.py` / `roihu_sna.py`-style implementations — a mechanical move with a real
risk, because those files are the ones Slurm jobs invoke by name and their CLI behaviour is what the
operators know.

## 2. Duplication inventory

**Identical function bodies in two modules (4 groups, all small helpers):**

| function | modules |
| --- | --- |
| `_text` | `ep24_graph_aggregate`, `ep24_mongo_export` |
| `_bounded` | `roihu_populism`, `step_7_roihu_discourse_network_analysis` |
| `_prompt_hash` | `roihu_populism`, `step_7_roihu_discourse_network_analysis` |
| `_clean_document` | `roihu_populism`, `step_7_roihu_discourse_network_analysis` |

The last three are one cluster: step 7 carries the discourse machinery that `roihu_populism` also
carries. It is the only consolidation in the repository that is unambiguously a copy — and it is safe
only after the shared behaviour is pinned, because two copies can already have diverged.

**Documented-intentional pair — do not merge without a decision:** the two Mongo exporters above.

**One duplicate pair, and both halves are live:** `ep24_logging.py` (204) is the
convention used by nine production modules (the five `puhti_*` and four `roihu_*`
stages); `ep24_logs.py` (163) is the same convention implemented again and is still
imported by `ep24_stage_orchestrator.py` (`import ep24_logs` at line 29,
`ep24_logs.configure_logging(` at line 722) as well as by `tests/test_ep24_logs.py`.
`tests/test_issue324_log_naming.py:4` records that one half of the log code "moved
into `ep24_logging` when the two halves were reconciled (#329)", but the orchestrator
was never migrated: deleting `ep24_logs.py` breaks the orchestrator's logging path.
This is a genuine duplicate to consolidate, not an orphan to remove.

## 3. Orphans — measured, not assumed

Modules with no importer, no reference in any `.py`/`.sh`/`.md`, and no CI reference:

| module | lines | last touched | evidence |
| --- | ---: | --- | --- |
| `ep24_legacy_clean.py` | 646 | 2026-10-01 | 0 occurrences anywhere; `ep24_cleaner` (526) and `roihu_clean` (1,100) are the two that are documented and tested |
| `roihu_storage_sync.py` | 62 | 2026-10-01 | 0 occurrences anywhere |
| `diagnose_ep24_step6.py` | 42 | 2026-10-13 | 0 occurrences anywhere; has a `main()`, so it may be a manual operator tool |

750 lines. (An earlier draft counted `ep24_logs.py` here; it is imported and called by
`ep24_stage_orchestrator.py`, so it is a duplicate, not an orphan — see §2.) Whether they are deleted or documented as tools is a
permission question (`AGENTS.md`), not a technical one.

**The cleaner cluster is the real triplication, and it needs a ruling rather than a delete:**
`ep24_cleaner.py` (526, described as "the public implementation" in
`docs/EP24_LEGACY_CLEANING.md:15`), `roihu_clean.py` (1,100, exercised by `tests/test_ep24_cleaning.py`
and cited in the country audits), `ep24_legacy_clean.py` (646, unreferenced). Three answers to "clean
the legacy per-country dataframe" is where a researcher currently has to guess.

## 4. Readability and explainability gaps

- **482 public module-level functions, 141 without a docstring (29 %).**
- **The five legacy stage modules have no module docstring at all** — and they are stage
  implementations, so that is precisely where a reader following "what happens in step 4" loses the
  thread. Three active stage modules also open without one: `roihu_summary`, `roihu_postprocess`,
  `roihu_populism`.
- **Entry points are not uniform**: eight are 10–137-line orchestrators, steps 7 and 8 carry 1,374 of
  the 1,634 entry-point lines. Stage bodies are 495–1,100 lines, which is where the "why" comments
  belong.
- **No prompt catalogue.** Prompt text lives inline in six stage modules (11 module-level prompt
  constants) with `ep24_prompt_contract.py` as the shared envelope. #411's deliverable 4 is therefore
  a new document, not a reorganisation.
- **Documentation collisions (narrow):** `EP24_LOGGING.md` (99) and `EP24_LOGGING_324.md` (43) both
  define the #324 log convention; `EP24_PREPROCESS_COMPATIBILITY.md` and
  `EP24_PREPROCESS_SCHEMA_COMPATIBILITY.md` both report #128. In each pair one should declare itself
  authoritative and the other link to it.

## 5. Plan, split by whether a row needs a ruling

### Can start immediately (documentation and measurement only)

1. **Repository map + this audit** — satisfies the "a researcher can locate each stage" criterion
   without touching code.
2. **Docstring pass on the active stages** (`roihu_summary`, `roihu_postprocess`, `roihu_populism`),
   then the legacy five: purpose, inputs, outputs, failure behaviour. Tests are the guard; no
   behaviour change.
3. **Declare the authoritative document in the two overlapping pairs**, and turn the other into a
   pointer.
4. **Prompt catalogue** (`docs/EP24_PROMPT_CATALOG.md`): for each of the 11 prompt constants — stage,
   task, inputs, output schema, version source, and whether the text is research instruction or
   schema boilerplate. Inventory only; prompts are not edited.

### Needs a ruling — blocked, and named

5. **Orphan removal** (§3): delete the 750 lines, or document them as manual operator tools?
6. **Cleaner cluster** (§3): which of `ep24_cleaner` / `roihu_clean` is canonical, and is
   `ep24_legacy_clean` a third implementation to retire or a framework to keep?
7. **`_prompt_hash` / `_clean_document` / `_bounded` consolidation** between `roihu_populism` and
   step 7 — safe only once a ruling says which module owns them.
8. **Extracting steps 7 and 8 into implementation modules** (§1). It is the largest structural win
   available and the most dangerous one: those two files are what Slurm invokes by name, so the move
   needs a compatibility entry point and a run-verified CLI, not just a green unit suite.

### Deliberately not proposed

- Merging the two Mongo exporters (documented as distinct contracts).
- Moving `puhti_*` to an archive directory: the tests, the compatibility rule and the published
  record all point at those paths. A move is a migration with downstream readers, not a tidy-up.
- Any prompt rewrite under this issue. #411 requires that prompt edits not silently change
  classification or interpretation criteria; the catalogue (item 4) is the precondition.

## What this pass did not do

No module was moved, renamed or deleted; no prompt, schema, CLI or output changed; the `legacy` branch
was not touched. This PR adds one document.
