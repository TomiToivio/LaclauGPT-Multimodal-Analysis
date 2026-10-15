# EP24 Steps 4–8 prompt/context contract consolidation (#241)

> **Status:** landed as a consolidation of four overlapping modules that this issue's own wave
> created (2026-10-10 … 2026-10-14, commits `ae669bb`, `5b2038b`, `d5e5fea`, `43a2271`, `b99b9b6`,
> `d7100a6`, `86a15e5`). Nothing human-coded, nothing on `legacy`, and no methodology change
> beyond one shared builder. Every budget delta is stated below; none of them is silent.

## Why there was something to consolidate

Four modules answered one question — what the model may see for Steps 4–8 — and the stages
imported different ones:

| module | lines | who imported it |
| --- | ---: | --- |
| `ep24_summary_evidence.py` | 159 | Step 4 (`roihu_summary.py`), the Step 4 coverage tool |
| `ep24_prompt_contract.py` | 282 | Steps 6, 7 and 8 (the cache fingerprint) |
| `ep24_evidence_packets.py` | 90 | Steps 6, 7 and 8 (the packet) |
| `ep24_context_budget.py` | 299 | Step 4, via `ep24_summary_evidence` |

Because each builder carried its own numbers, a stage's effective budget depended on which builder
it happened to import. The runbook's table and `ep24_prompt_contract`'s post-reserve defaults
disagreed for Steps 4, 7 and 8. That is the defect this change removes.

## The canonical module

**`ep24_prompt_contract.py` is the single source.** Evidence for choosing it:

1. Three of the four stages (6, 7, 8) already imported it in production for the cache fingerprint,
   so it is already the module the majority of Steps 4–8 depend on. Choosing it leaves
   `effective_prompt_fingerprint` — the function #287 corrected — exactly where it is.
2. Its own commit declares it the shared model-visible prompt/context contract for Steps 4–8;
   the runbook and the issue's acceptance criteria are all contract-shaped (budget, provenance
   labels, per-field truncation accounting, fingerprint), which is this module's stated scope.
3. The other three are a Stage-4 adapter (`ep24_summary_evidence`), a Stage-4 budget policy
   (`ep24_context_budget`) and a Steps 6–8 field table (`ep24_evidence_packets`).

`ep24_context_budget` had the richer accounting (`label` distinct from field name, `omitted` with a
reason, `render_report`, `summarise`). That part was worth keeping, so the canonical module adopted
its vocabulary rather than the thinner one it had. What did **not** survive is
`ep24_prompt_contract`'s aspirational `stream_caps` for Steps 6–8 (`speech`, `signs`, `utterances`,
`quotes`): no stage ever used those stream names, and the real per-field caps live in the Steps 6–8
field tables.

## Effective model-visible budget, before and after

Measured by building a packet for a row with every field oversized, through each stage's *real*
production entry point (`roihu_summary.build_stage_packet(4, …)`, `roihu_populism.build_prompt_context`,
`step_7_….build_prompt_context`, `step_8_….sna_evidence_report`).

| stage | before: effective cap | before: packet realised | after: effective cap | after: packet realised | delta |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 17,200 | 17,198 | **20,000** | 20,000 | **+2,800** |
| 6 | 28,000 | 27,969 | 28,000 | 28,000 | cap unchanged |
| 7 | 20,000 | 19,968 | 20,000 | 19,992 | cap unchanged |
| 8 | 12,000 | 11,969 | 12,000 | 12,000 | cap unchanged |

The contract values are the runbook's documented numbers, which are what the stages were specified
against: Step 4 20,000 / Step 6 28,000 / Step 7 20,000 / Step 8 12,000 model-visible characters.

### Delta 1 — Step 4 gains 2,800 model-visible characters

Before, `ep24_summary_evidence` passed a **total** of 20,000 to `ep24_context_budget`, which
subtracted a 2,000-character output reserve and an 800-character schema reserve, leaving 17,200 for
input. After, the runbook's 20,000 is the model-visible budget itself, and the reserve is recorded
as headroom the stage's context window must also hold (`StageBudget.declared_total_chars` = 22,800
for Step 4). The contract number is the right one because the runbook — the document the 10-per-
country benchmark is specified against — states the per-stage figure as a **model-visible** budget,
and Step 6 already agreed with it on exactly that reading.

Consequence: `LACLAUGPT_SUMMARY_EVIDENCE_CHARS` now sets the model-visible budget directly. It was
already being set to 20000 by `scripts/ep24/test_step4_fi.sh`, which is the model-visible number
the runbook documents, so the fixture and the contract now agree instead of differing by 2,800.
`LACLAUGPT_SUMMARY_RESERVE_OUTPUT_CHARS` / `LACLAUGPT_SUMMARY_RESERVE_SCHEMA_CHARS` still resolve and
are still validated (both must be non-negative and must fit inside the model-visible budget), but
they no longer shrink the packet a second time.

### Delta 2 — Steps 6–8 now fill their documented budget

The previous Steps 6–8 builder reserved an unexplained 44-character slack
(`remaining = budget - len(joined) - len(heading) - 44`), so on an oversized row ~24–31 characters
of the stage's cap went unused. The canonical builder fills the documented budget exactly. The
**cap** is unchanged for all three stages; only the last field on a truncated row is ~24–31
characters longer. No field cap changed for any stage.

### Delta 3 — one section format for all four stages

Every stage now emits one heading shape: `[<column> | <provenance label>]`. Steps 6–8 already used
it, so their prompt text is unchanged apart from Delta 2. Step 4 used `### <provenance label>`, so
its headings change shape while keeping the same label text — the provenance statement the model
reads is identical, and the field's column name is now visible next to it. Step 4's truncation
marker also becomes the short ` [TRUNCATED]` form Steps 6–8 already used.

Step 4's prompt text therefore changes, and so does the Step 4 cache key built from it. That is the
designed behaviour, not a side effect: see below.

### Delta 4 — a below-floor operator override now fails loudly

Step 8 used to clamp `LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS` to 3,000 (`max(3000, env)`); the contract
now refuses a value below the stage's floor with a `BudgetError` naming the variable. A value of
0–2,999 is not a usable budget for a JSON-schema-constrained extraction, and silently substituting
a different number is exactly the class of quiet divergence this issue is about. Step 4 refused an
out-of-range budget before this change too; Step 8 is now consistent with it.

### Not a delta — per-stage field tables, order and caps

The Step 4 field order, labels, caps (`LACLAUGPT_SUMMARY_CAP_<STREAM>`) and metadata block are
identical. The Steps 6–8 field lists, their order, their per-field caps and their provenance role
strings are copied across unchanged, including the deliberate `asr_transcript` cap difference
between Steps 6–7 (9,000) and Step 8 (5,000). Steps 6–8 still treat an empty JSON container
(`[]`, `{}`) as no value, and Step 4 still keeps it, exactly as before.

## Cache fingerprints invalidate correctly

The #287 fingerprints cover prompt, model **and** generation options. They are built over the
prompt and the packet text:

* Step 4: `_prompt_sha256(system_prompt, user_prompt, model)`, whose payload already includes
  `_summary_options()` (`num_ctx`, `num_predict`, `temperature`). Unchanged by this consolidation —
  it covers the same four components as `effective_prompt_fingerprint`, so there was nothing to fix.
* Steps 6 and 7: `effective_prompt_fingerprint(SYSTEM_PROMPT, f"{PROMPT_VERSION}\n{context}", model,
  {temperature, num_ctx, num_predict})`.
* Step 8: `effective_prompt_fingerprint(EXTRACTION_SYSTEM, f"{PROMPT_VERSION}\n{packet}", model,
  {temperature, num_ctx})`.

Every one of them hashes the packet text, so a packet this change renders differently cannot collide
with an answer produced from the old one: the new run misses the cache and recomputes, and no stored
result is reused as if it were current. `tests/test_prompt_contract.py` pins this property
(`test_a_changed_packet_cannot_reuse_the_fingerprint_of_the_old_one`) alongside the prompt/model/
option sensitivity tests. `PROMPT_VERSION` strings were deliberately left alone: the fingerprint
already changes through the context component, and editing the version constants would be a second,
redundant signal.

Expected consequence for Step 4's durable caches: `context_sha256` values change, so the SQLite
cache and the MongoDB provenance key miss once and the affected records recompute. Steps 6–8
recompute only where a packet actually changed, i.e. rows whose last field was being clipped.
No stored row is rewritten, and the source CSVs and MongoDB collections are not touched by this
change at all (it adds and removes no columns; `sna_context_truncated` gains one additive JSON key,
`model_visible_chars`).

## What was removed, and what replaced it

| file | change |
| --- | --- |
| `ep24_prompt_contract.py` | **the canonical contract.** Gains the real per-stage field tables, `build_packet`, `build_stage_packet`, `Packet`/`FieldReport`, `stage_budget`, the operator env vars, and the budget/reserve vocabulary. `effective_prompt_fingerprint` and `estimate_tokens` are unchanged. |
| `ep24_context_budget.py` | **removed.** Its builder, accounting and `clean` moved into the canonical module; its dead generic `LACLAUGPT_CONTEXT_CAP_*` env was not carried over. |
| `ep24_evidence_packets.py` | **removed.** Its Steps 6–8 field tables and role strings moved into the canonical module as `STEP67_*`, `STEP678_*`, `STEP7_*`, `STEP8_*` and `role_for`. |
| `ep24_summary_evidence.py` | **removed.** Its Step 4 labels, caps, metadata keys, `choose`, the retrieval opt-in and the empty-JSON-container decision moved into the canonical module as `step4_streams`. |
| `tests/test_ep24_context_budget.py`, `tests/test_ep24_evidence_packets.py`, `tests/test_summary_evidence.py` | **removed** with their modules. |
| `tests/test_prompt_contract.py` | rewritten as the one contract suite; nothing the three removed suites covered is dropped. |
| `roihu_summary.py` | Step 4 builds its packet with `build_stage_packet(4, row, memory=…, rag=…)`. |
| `roihu_populism.py` | Step 6's `build_prompt_context` uses the canonical builder (signature and defaults unchanged). |
| `step_7_roihu_discourse_network_analysis.py` | Step 7's, likewise. |
| `step_8_roihu_social_network_analysis.py` | Step 8 builds and audits from one packet; the per-field report now comes from the packet's own structured accounting instead of re-parsing its text with a regex. |
| `scripts/ep24/step4_evidence_coverage.py` | imports the builder and `clean` from the canonical module. |
| `tests/test_roihu_memory_codebooks.py` | the Step 6 guard now asserts on the packet the builder produces instead of on another module's source text. |

Steps 2 and 5 are **out of this issue's scope** (its goal text names Steps 4–8) and keep their own
stage-local bounds, `LACLAUGPT_FRAME_CONTEXT_MAX_CHARS` and `LACLAUGPT_STEP5_MAX_PROMPT_CHARS`, with
their existing tests. They were not migrated here and should not be read as covered by this contract.

## Invariants preserved

* MongoDB and the cumulative CSV remain the source of truth; no storage, schema or field meaning is
  touched, and no source information is deleted — truncation is marked in-band and reported per
  field, and the caller's row is never mutated (pinned by tests for all four stages).
* Memory, corpus RAG and other stages' outputs are never evidence: Steps 6–8 do not read those
  columns at all, and Step 4 takes them only behind `LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL=1`, as
  `CONTEXT`, labelled `NOT video evidence`, and after every evidence field. No stage's own earlier
  output is fed back to it.
* The EP24 first-1.0-second scroll invariant is untouched and is still imported from
  `ep24_video.py`; no video, ASR, frame or OCR module changed.
* The additive-additive field rule holds: no column was added or removed.
* `LACLAUGPT_MAX_ROWS=100`, the numbered 1–9 batch contract, Step 3's reservation, and the `legacy`
  branch are untouched. The Leifeld DNA schema, the locked Laclau/Palonen prompts, Step 8's graph
  invariants and Step 9's determinism are unchanged. Prompts were not rewritten; only the packet
  layout the prompts receive changed, and only where stated above.

## Still open

Nothing in this document can substitute for the runbook's manual sample review
(`## Benchmark before reruns`): 10 source IDs each for Finland, Poland and Portugal, scored by hand
against the source. It needs the runtime and the human, and this change does not claim it. It is
the acceptance criterion #241 keeps open.
