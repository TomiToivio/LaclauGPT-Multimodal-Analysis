# Refactor audit — repository readability, duplication and explainability (#411)

Status: **before/after audit, deliverable 1 of #411.** Nothing in this document changes the pipeline.
Every count was produced by reading the repository at `858f383`, not by memory; the script that produced
them is described at the end so the audit can be repeated.

This is the *before*. A refactor that starts by rewriting stages without this list produces a tidier
repository that has lost the reason it looked untidy.

## 1 Shape of the repository

| group | files | lines | what it is |
| --- | ---: | ---: | --- |
| `tests/` | 179 | 23,652 | pytest suite, roughly one test module per issue |
| `ep24_*` | 37 | 11,993 | orchestration, contracts, export, audits, storage helpers |
| `roihu_*` | 15 | 9,398 | stage implementations and codebook/memory/RDF support |
| `step_*` | 10 | 1,654 | the executable stage entry points |
| other (scripts, examples, experiments, backends) | 50 | ~8,700 | task tools, demos, ASR/OCR adapters, scratch |
| **total** | **292** | **~57,400** | |

Readability measurements:

* **99 files carry an `if __name__ == "__main__"` block** — a researcher looking for "the entry point" has
  99 candidates before reading `ep24_stage_contract.py`.
* **39 modules have no module docstring**, including three of the nine stage implementations:
  `roihu_summary.py` (741 lines), `roihu_postprocess.py` (623), `roihu_populism.py` (618).
* **81 Markdown documents** in `docs/`, several of them audits of the same subsystem from different issues.
* Longest functions: `roihu_summary.analyze_videos` (301 lines), `ep24_stage_orchestrator.run_country`
  (301), `roihu_postprocess.analyze_responses` (286), `roihu_preprocess.preprocess_dataframe` (282),
  `experiments/vllm_video_test.main` (322).

## 2 Duplication inventory, with a verdict for each

The question for every pair below is not "do these look alike" but **"is the duplication deliberate?"** A
refactor that cannot tell the difference deletes the wrong one.

### 2.1 Consolidate — two implementations of one decision

**`ep24_logging.py` (204 lines) vs `ep24_logs.py` (163 lines).** Both carry the same docstring — *"One log
directory, one filename convention for every EP24 pipeline step (#324)"* — and both implement the same
concept with different spelling: `normalise_step`/`normalize_step`, `log_path`/`canonical_log_path`,
`config()` vs `configure_logging()`. `git log` shows they arrived as two PRs for the same issue (#326 and
#329). This is not a convention with two faces; it is one decision implemented twice, and a caller cannot
know which is authoritative.

*Usage, which decides the survivor:* `ep24_logging` is imported by 11 modules, `ep24_logs` by 2
(`ep24_stage_orchestrator.py` and its test). But the orchestrator uses `start_run()`/`open_log()`, which
`ep24_logging` does not have — so the merge is "move the two functions the orchestrator needs into
`ep24_logging`, then reduce `ep24_logs` to a re-export shim with a deprecation note". Verdict:
**consolidate, one PR, with `tests/test_ep24_logs.py` kept and extended.**

### 2.2 Consolidate — overlapping export tools

`ep24_mongo_export.py` (845 lines, imported by 14 modules) and `ep24_mongo_dashboard_export.py` (487 lines,
imported by one test and nothing else) both describe themselves as the read-only EP24 export from shared
MongoDB (#306). The dashboard variant is dead weight on the import graph but may still be the shape a
dashboard consumes. Verdict: **check the dashboard consumer first, then either merge the shaping into
`ep24_mongo_export` as an option or archive the module with a documented reason.** Not urgent; not safe to
delete blind.

### 2.3 Consolidate — three coverage audits, five export audits

Country × stage coverage is implemented three times: `ep24_coverage_audit.py` (195 lines, 3 importers),
`scripts/ep24/audit_stage_coverage.py` (183), `scripts/ep24/audit_country_stages.py` (214). Export auditing
appears six times in `scripts/ep24/`: `audit_export.py` (602), `audit_export_bundle.py` (488),
`audit_export_contract.py` (128), `audit_export_diff.py` (189), `audit_ep24_stages.py` (168),
`audit_issue405_sources.py` (145).

These are not one decision implemented six times — each answers a different question (contract, bundle
integrity, diff between two exports, per-country matrix, aggregate QA). What they duplicate is **plumbing**:
argument parsing, reading a manifest, printing a table. Verdict: **keep the tools, extract one shared
`scripts/ep24/_audit_common.py` for manifest reading and report printing.** Low risk, immediate payoff,
and it makes the six tools comparable at a glance.

### 2.4 Keep and document — the historical implementations

`puhti_*.py` (5 modules, 1,332 lines) are near-parallels of `roihu_frame/summary/postprocess/populism/preprocess`.
This is deliberate: `AGENTS.md` records that published results depend on the historical implementation, and
`tests/test_legacy_safety.py` pins the correspondence between the two families so neither can drift
unnoticed.

Verdict: **do not consolidate, do not delete, and — the part that is missing — do not let them be
confusing.** Their five modules have no docstring, so a reader meets `puhti_summary.py` with no statement
of why it exists. The fix is one paragraph at the top of each, plus a row in `docs/REPOSITORY_MAP.md`
(already added). That is explainability at the cost of five docstrings, not a refactor.

### 2.5 Keep — duplicates that are the method

`roihu_clean.py` (1,100) / `ep24_cleaner.py` (526) / `ep24_legacy_clean.py` (646) share function names
(`clean_dataframe`, `write_outputs`, `render_markdown_report`). They clean different things — legacy
per-country frames, legacy exports, and the active pipeline's input — and the shared names describe a
shared *shape*, not shared behaviour. Verdict: **keep; rename only if a reader is genuinely misled**, which
the audit could not establish.

`scripts/ep24/apostrophe_hygiene.py` and `cyrillic_hygiene.py` share two small helpers, and their
docstrings explain a real design difference (French elision vs Latin/Cyrillic homoglyphs). Verdict: **keep;
optionally share `_iter_labels`/`_candidate_files`**, which are the only genuinely identical bodies found in
the whole repository (2 pairs).

### 2.6 Archive candidates

* `experiments/` — 1,519-line `vllm_video_test.py` plus scratch scripts. One file is load-bearing (step 3
  wraps it); the rest is not pipeline code and should not be read as if it were. Verdict: **document the
  boundary** (done in the map), do not move it in this issue.
* `diagnose_ep24_step6.py` (43 lines) — fine where it is; listed so it is not mistaken for a stage.

## 3 Explainability gaps, ranked by what a researcher hits first

1. **The three biggest stage implementations have no module docstring.** `roihu_summary`,
   `roihu_postprocess`, `roihu_populism` are 2,000 lines of the most analytically consequential code in the
   repository; a reader has to infer their role from column names. (Highest value, lowest risk: prose only.)
2. **No single document says which module is authoritative.** Added: `docs/REPOSITORY_MAP.md`, generated from
   `ep24_stage_contract.py` for the stage table and checked by a test so it cannot rot.
3. **Prompt provenance is spread across `ep24_prompt_contract.py` and per-stage builders.** A reader can find
   the prompt text but not a catalogue of *every* prompt with its inputs, outputs and version. This is
   deliverable 4 of #411 and is not done yet.
4. **The stage contract's `notes=` carry the analytical role, and they are excellent — but they live in
   Python.** The map now surfaces them; the next step is a generated document so they are readable without
   reading code.
5. **99 `__main__` blocks.** Most are legitimate CLI conventions; the absence of a stated entry-point policy
   is what makes the count alarming. The map states the policy: `step_N_*` runs, `roihu_*` implements,
   `scripts/` does tasks.

## 4 Prioritised plan, in the order that makes each step safe

| # | increment | risk | why here |
| --- | --- | --- | --- |
| 1 | `docs/REPOSITORY_MAP.md` + the ratchet test | none | A reader needs the map before any code moves; the test stops it rotting. **Done in this PR.** |
| 2 | Module docstrings for the three undocumented stage implementations | none (prose only) | The highest-value explainability gap, and it cannot break behaviour. |
| 3 | Consolidate `ep24_logging`/`ep24_logs` | low | One decision, two implementations; the test suite already covers the surviving path. |
| 4 | `scripts/ep24/_audit_common.py` | low | Six tools, one plumbing duplication. |
| 5 | Generated stage guide from `ep24_stage_contract.py` | none | Turns the `notes=` field into researcher-facing documentation without copying it. |
| 6 | Prompt catalogue (deliverable 4) | none | Requires reading every prompt; do it after the stages are documented. |
| 7 | Dashboard export consolidation | medium | Needs a consumer check first. |
| 8 | Stage-internal readability (the four 280–320-line functions) | medium | Function-level extraction changes behaviour surfaces; do it last, one function per PR, with the existing tests as the safety net. |

Deliberately **not** in the plan: renaming modules, moving files, or introducing a package layout. All three
would break the Slurm wrappers, the `step_*` import conventions and every documented command, for a benefit
that documentation delivers without them.

## 5 What this audit does not claim

* It is static analysis plus reading, not profiling: a module that is "imported by one test" may still be
  used by an operator from a shell, and the audit cannot see that.
* The duplication verdicts in §2.1–2.3 are judgements, stated with the evidence that grounds them. The ones
  that would delete code (§2.2) are marked "check the consumer first" for that reason.
* No behavioural comparison is reported here, because this PR changes no behaviour. Every later increment in
  §4 states its own verification.

## Reproducing the numbers

The counts in §1–§2 come from a read-only AST pass over the repository: per-file line counts and docstring
presence, function lengths, function-name collisions across modules, identical function bodies (whitespace
normalised, ≥400 characters), and `import`-graph fan-in per module via `grep -c`. Any of these can be
re-derived with `python -c` in a minute; none of them depends on a database or a cluster.
