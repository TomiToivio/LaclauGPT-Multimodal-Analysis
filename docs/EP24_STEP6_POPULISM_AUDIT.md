# Step 6 `roihu_populism.py` — audit and refactor plan (issue #170)

Status: **audit complete; modern Step 6 implemented and hardened.** This document
began as the mandated first step of #170 ("First audit the current step very
carefully"). The findings below describe the legacy implementation that motivated
the refactor. The active `roihu_populism.py` now uses evidence-linked structured
coding, explicit abstention, shared Mongo/context infrastructure, Redis
coordination, cumulative CSV preservation, and deterministic legacy projections.

Scope of the review: correctness, hidden bugs, exception handling, retry/restart
semantics, logging, output validation, schema compatibility, cache correctness,
duplicate processing, country handling, model configuration, prompt size,
invalid/partial LLM JSON, silent fallbacks, prior-column preservation, and
downstream DNA/SNA/RDF interaction.

---

## 1. Executive summary

`roihu_populism.py` is a single-shot script, not a pipeline stage. It is missing
every integration the current EP24 architecture provides (MongoDB, RAG, memory,
Redis, codebook-as-context, cumulative-row preservation), and its three most
severe defects compound into **silent data loss**:

1. it rewrites the whole CSV **inside the row loop**, unguarded against a
   truncated input (BUG-1);
2. it **swallows every LLM/parse failure** and proceeds with empty outputs
   (BUG-2);
3. it has **no superset guarantee** on the output schema (BUG-7).

Together: one transient LLM error mid-run can write a truncated, partially-empty
CSV to the canonical output path. The theoretical prompt is also several
generations behind `THEORY.md` (presupposes populism, mechanically maps affect,
equates frontier with "enemy"). Section 5 lists the theory corrections; section 6
the downstream contract that must be preserved; section 7 the staged plan.

---

## 2. Structural problems

| # | Location | Problem |
|---|---|---|
| S-1 | whole file | No `ep24_cli` wrapper; the numbered-stage CLI contract is absent. `DATASET_COUNTRY` is language→country, but the pipeline selects by **country**. |
| S-2 | 20–21, 464–466 | SQLite connection opened at **import**; closed only under `__main__`. Importing the module creates `formula_of_populism.db` in the CWD. |
| S-3 | 16 | `logging.basicConfig(...)` is a no-op if logging was already configured (library import), so `formula.log` may never be written; `formatter` (17) is dead code. |
| S-4 | 322–325, 354–359, 400–427 | The full system prompt is `logger.info`-ed and `print`-ed on import; responses are `print`-ed repeatedly. Issue §14 forbids this by default. |
| S-5 | 336–363 | Three different return types from `get_response` (`FormulaOfPopulism`, raw `str`, `{}`) under one declared contract. |
| S-6 | 30–35 | Country mapping is language-keyed and has no `en`; inconsistent with `ep24_cli.COUNTRY_TO_LANGUAGE`. |
| S-7 | 1 | `from os import name, system` — `system`/`name` unused. |

---

## 3. Hidden bugs

Severity: **C**ritical (data loss / wrong results), **H**igh, **M**edium, **L**ow.

### BUG-1 — C — full-CSV rewrite inside the row loop, unguarded vs truncation
Lines **425** (miss branch) and **447** (hit branch) call `df.to_csv(output, index=False)`
on the in-memory frame, once per successfully processed row.

`LACLAUGPT_MAX_ROWS` truncates the frame at **370–373** (`df = df.head(max_rows)`),
but the output path is `LACLAUGPT_OUTPUT_CSV or filename`. When the demo limit is
active and the output path is the full file, the first row write **overwrites the
canonical CSV with only the head rows**. This is the most dangerous defect in the
file. Contrast Step 4 (`roihu_summary.py`), which snapshots `before`, writes
through `write_cumulative_csv(before, df, output)`, and checkpoints deliberately.

### BUG-2 — C — LLM/parse failures are swallowed; processing continues
`get_response` catches `Exception` and returns a bare dict (337, 360–362). The
caller then reads `response.populism_analysis` (402–404). Consequences:

- request failure → `{}` → `KeyError`/`AttributeError` in the caller;
- schema-validation failure (356) → `llama_response` is left as the **raw string**
  → `.populism_analysis` on a `str` raises;
- both are caught by the per-row `except` (448–450), logged, and the row is left
  with **empty** Step 6 columns.

No raw response is retained, no retry, no error status recorded. Issue §9/§16
require retaining raw output and metadata on parse failure.

### BUG-3 — H — cache key is `video_file`, not the stable record ID
Line **385** builds the key as `video_filename` → `allas_filename` → `stable_source_id(row)`.
The key is **not** the pipeline's stable ID, omits `country` and `video_id`, and
changes meaning when `allas_filename` differs or is absent. `stable_source_id`
(which exists for exactly this purpose) is only a last-ditch fallback. Cross-run
and cross-country key drift are possible.

### BUG-4 — H — cache hit drops `laclau_summary_md`
The miss branch sets `laclau_summary_md` (423); the hit branch (429–447) restores
only `analysis`/`us`/`frontier`. A rerun that hits the cache therefore **loses
`laclau_summary_md`** for those rows, leaving the column inconsistent between
cached and fresh rows — and it is a declared Step 6 output
(`ep24_stage_contract.py:145-147`).

### BUG-5 — H — cache is not invalidated by prompt or model change
The cache stores three text fields keyed by `video_file` only. Changing the system
prompt or the model **silently reuses stale results**. Step 4 solved this by
including a context hash in the key; Step 6 has no prompt/model/version in its key.

### BUG-6 — H — no output-superset guarantee
`ensure_columns` (374) adds Step 6 columns; nothing asserts the output contains
**every** input column. There is no `before` snapshot and no verification. Combined
with BUG-1 (immediate full rewrites), any accidental column loss propagates at once.
Issue §2/§13 forbid this.

### BUG-7 — M — hardcoded, small context window
**340–345**: `num_ctx=8192`, `num_predict=2048`, not configurable. The prompt is
`metadata_context(row) + summary_analysis` (392), i.e. the smallest possible
context, and 8192 is too small for a full cumulative row. Issue §2 requires a
deterministic, budgeted, provenance-labelled context; other steps read these from
environment variables.

### BUG-8 — M — duplicate-concurrency unsafe; no per-record lock or status
There is no Redis coordination (issue §7) and no per-record lock, so two
concurrent runs can both miss the cache and double-process the same rows.

### BUG-9 — M — `ensure_columns` on `df` may not preserve a cached row's other fields on restart
Restart relies entirely on the SQLite cache plus a full-frame rewrite; because the
cache stores only 3 fields (BUG-4/BUG-5) a restart cannot reconstruct the complete
row for cached records from the cache alone.

### BUG-10 — L — dead/unused code
`global system_prompt` (366) is unused; `json` (5) is used only in the codebook
path; `formatter` (17) is never attached.

---

## 4. Integration gaps vs the EP24 architecture

| Issue § | Requirement | Current state |
|---|---|---|
| §3 | MongoDB via `ep24_db.country_storage`, full cumulative document, safe upsert | **absent**; local SQLite only |
| §4 | Memory via `ep24_memory` (`normalization_context_not_source_evidence`) | **absent** |
| §5 | RAG via `ep24_rag`, context marked `prior_analysis_context_not_source_evidence`, retrieval + post-stage upsert | **absent** |
| §6 | Versioned country-codebook retrieval, selected entries + fingerprint recorded | opt-in, whole-`context_block` append, not versioned into the row |
| §7 | Redis via `ep24_redis`, optional | **absent** |
| §13 | Mongo + CSV backup, restart reconstructs the **full** row | SQLite is an independent source of truth |

---

## 5. Theoretical deviations from `THEORY.md` (the core of #170 §8/§9/§10)

| # | Deviation | Current code | Required (THEORY.md) |
|---|---|---|---|
| T-1 | **Presupposes populism** | 77/83/96/101; "Explicitly restate the **populist** discourse" (140); examples all populist | §6.4 **Abstention**: allow no-Us, no-Frontier, empty chains |
| T-2 | **Mechanical affect polarity** | "people = demands + **positive** emotions, frontier = … + **negative** emotions" (90); formula `[Positive Affects] + Frontier … [Negative Affects]` (143) | §6.3 affect is **not** detachable sentiment; do **not** map Us→positive / Frontier→negative |
| T-3 | **Frontier ≡ enemy** | "antagonistic other" (119), "symbolic enemies" (172) | §6.2 Frontier is a constructed boundary; distinguish opposition / criticism / blame / threat / exclusion / boundary / antagonism (AI26 `FrontierConstruct.relation`, ref lines 59–66) |
| T-4 | **No evidence discipline** | `FormulaOfPopulism{analysis:str, us:[{element,affect}], frontier:[…]}` (327–334) | §13.1 spans, confidence, provenance, uncertainty, counter-evidence (AI26 has all four) |
| T-5 | **Affect reified as a one-word label** | "single-word emotional labels" (190); `populism_affect: str` | §6.3 affect is relational; carry target + evidence |
| T-6 | **No signifier safeguards** | prose mentions chains/empty signifiers (128–130) but the model has no field and no rule against co-occurrence-as-chain | §3/§4; AI26: "frequency is not hegemony", "polysemy alone is not floating/empty signification" |
| T-7 | **Single call → theoretical fact** | `populism_analysis` free text is the only output (228) | §13.1 candidates only; "no automatic final theoretical fact from one LLM call" |
| T-8 | **No corpus-level guard** | none | §7/§9: hegemony and polarisation are **corpus-level**; one video cannot establish them |

---

## 6. Downstream compatibility that MUST be preserved

These are hard constraints; the refactor must keep them, derived deterministically
from the new structured output (issue §10 Compatibility):

1. **RDF contract** — `roihu_csv_rdf.py:90-105` parses `formula_of_populism_us` and
   `formula_of_populism_frontier` as **one `element^affect` pair per line**, exactly
   one `^`, non-empty parts; anything else is retained as a raw cell with a warning.
   The compatibility serialization must emit exactly this format.
2. **Declared Step 6 outputs** — `ep24_stage_contract.py:145-147`:
   `formula_of_populism_analysis`, `formula_of_populism_us`,
   `formula_of_populism_frontier`, `laclau_summary_md`.
3. **Cleaner/legacy cleaners** — `roihu_clean.py:81-87`, `ep24_cleaner.py:40` also
   recognise `formula_of_populism_us_elements`, `_frontier_elements`,
   `_us_affects`, `_frontier_affects`.
4. **Tests** — `tests/test_ep24_cleaning.py:45-51`, `tests/test_legacy_pipeline_contract.py:71`,
   `tests/test_roihu_csv_rdf.py:24-25`.

New richer structure is **added** as new JSON columns; the four legacy columns above
remain, derived deterministically.

---

## 7. Implementation plan (staged, each step independently testable)

| Step | Deliverable |
|---|---|
| S1 | `build_step6_context(row)` — deterministic, budgeted, four provenance classes (**current evidence** / **derived prior analysis** / **researcher+codebook memory** / **retrieved corpus context**); log+hash the effective context. |
| S2 | Modern Pydantic schema (`UsConstruct`, `FrontierConstruct`, `AffectObservation`, `FormulaComponents` + EP24 discourse fields) with spans, confidence, provenance, uncertainty, counter-evidence, `prompt_version`, model metadata; **deterministic** legacy serializer producing the `element^affect` lines. |
| S3 | Prompt ported from the AI26 architecture to EP24 theory: abstraction permitted, no forced Us/Frontier/affect, disagreement ≠ antagonism, sentiment ≠ affective investment, signifier safeguards, no per-document hegemony/polarisation, Palonen dynamics as provisional. |
| S4 | Persistence: Mongo via `country_storage` (patch semantics preserving prior fields, stable `_storage_id`), Redis opt-in via `RedisCoordinator`, memory via `ep24_memory`, RAG via `ep24_rag` (+ `update_retrieval`), versioned codebook retrieval; SQLite reduced to a **subordinate** cache keyed by `stable_source_id` + context/prompt/model hash. |
| S5 | CSV checkpointing mirroring Step 4 (single end-of-run write, deliberate intermediate checkpoints, **never** head-truncating the canonical path). |
| S6 | CLI via `ep24_cli`; env-configurable `num_ctx`/`num_predict`; raw-response retention on parse failure; error status + timestamps; reproducibility logging. |
| S7 | Tests per issue §16 (theory safeguards, structured output, pipeline contract, context systems). |
| S8 | Integration runs: **Finland → Poland → Portugal**, then the remaining countries. |

---

## 8. Resolved design decisions

1. **No permanent/binary actor label.** Step 6 remains candidate/evidence based.
   It exposes `formula_minimum_conditions_met` as a document-level evidentiary
   condition, not a populism score or permanent party/actor classification. A
   mechanical guard forces it false unless both a collective Us and an explicit
   `antagonistic_frontier` are present in the structured result.
2. **Legacy `element^affect` projection never fabricates affect.** A candidate
   without an evidenced target-linked affect remains fully available in the rich
   JSON but is omitted from the historical compatibility column. This keeps RDF
   syntax valid without converting "not evidenced" into a fake emotion.
3. **MongoDB is durable restart state.** Step 6 resumes a completed record only
   when stable record ID, prompt version, model and prompt/context hash match.
   There is no independent SQLite source of truth.
4. **No circular rerun context.** Existing Step 6-owned `laclau_*` and
   `formula_of_populism_*` outputs are excluded from the next Step 6 prompt, so
   a rerun cannot reinforce its own prior interpretation and the resume hash stays
   stable.
5. **Sample safety.** A limited run refuses to overwrite its own input path when
   the limit would truncate the dataframe; samples must write to a distinct
   checkpoint/output path.
