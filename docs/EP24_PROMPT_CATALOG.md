# EP24 prompt catalogue (#411, deliverable 4)

**Status:** inventory only. Measured on `main` at `7a03afc`. No prompt text, prompt version,
schema or stage behaviour is changed by this document. It exists so that the next change to a
prompt is a reviewed decision instead of an edit inside a 700-line stage module.

**Why this is a separate document:** #411 requires a catalogue of every prompt with its stage,
task, inputs, output schema and version, and requires that prompt edits never silently change
classification or interpretation criteria. Both are impossible while the twelve prompts live
inline in eight modules — seven stage or enrichment modules and one experiment harness.

**Related:** [`EP24_PROMPT_CONTRACT_CONSOLIDATION.md`](EP24_PROMPT_CONTRACT_CONSOLIDATION.md)
(the shared envelope and the per-stage budgets), [`EP24_PROMPT_EVIDENCE_RUNBOOK.md`](EP24_PROMPT_EVIDENCE_RUNBOOK.md)
(the evidence and rerun gates), [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) §4.

---

## 1. Registry

Eight active prompts, four frozen legacy prompts. "Budget" is the **model-visible** character
budget enforced by `ep24_prompt_contract.STAGE_BUDGETS` (Steps 4, 6, 7, 8) or by the stage's own
constant (Steps 2, 3, 5).

| # | Stage | Prompt symbol | Version constant | Output | Budget | Retries |
|---:|---|---|---|---|---:|---:|
| 1 | 2 frame | `get_analysis()` — `system_prompt` + `instruction` | `FRAME_PROMPT_VERSION` | free text | `num_ctx` 8192 | none |
| 2 | 3 video | `SYSTEM_PROMPT` + `VIDEO_PROMPT` | `PROMPT_VERSION` | `analysis_markdown` + `SCROLL` JSON | vLLM context | guided-JSON regen |
| 3 | 4 summary | `get_llama_summary_system_prompt()` + `get_llama_summary_user_prompt()` | **none** | free Markdown | 20,000 | 2 (empty/incomplete) |
| 4 | 5 postprocess | `get_system_prompt()` + `build_postprocess_prompt()` | **none** | `PostprocessResult` | 12,000 (own cap) | none |
| 5 | 6 discourse | `SYSTEM_PROMPT` + retry suffix | `PROMPT_VERSION` | `EP24DiscourseAnalysis` | 28,000 | 3 |
| 6 | 7 DNA | `SYSTEM_PROMPT` + retry suffix | `PROMPT_VERSION` | `DNAResult` | 20,000 | 3 |
| 7 | 8 SNA | `EXTRACTION_SYSTEM` | `PROMPT_VERSION` | `SNAResult` | 12,000 | 3 |
| 8 | enrichment (between 5 and 6) | `ollama_adjudicator()` — fixed system line + JSON payload | **none** | `{"entity_id": str}` | payload only | none |
| 9 | 2 legacy | `get_analysis()` | none | free text | n/a | none |
| 10 | 4 legacy | `get_llama_summary_system_prompt()` | none | free Markdown | n/a | none |
| 11 | 5 legacy | `get_system_prompt()` | none | free text + JSON block | n/a | none |
| 12 | 6 legacy | module-level `system_prompt` (line 28) | none | free text + JSON block | n/a | none |

Steps 1 (ASR/OCR) and 9 (RDF) send no text to a language model. The enrichment prompt (#8) is
not part of the numbered chain: `roihu_enrich.py` runs it only when
`LACLAUGPT_ENTITY_LLM_ADJUDICATION` is enabled.

## 2. Active prompts

### P1 — Step 2, single-keyframe analysis

- **Where:** `roihu_frame.py`, inside `get_analysis()`: the system text at line 147 (890
  characters) and the output-heading instruction at line 161 (540 characters).
- **Version:** `FRAME_PROMPT_VERSION = "ep24-frame-v3-placeholder-scope-20261011"`.
- **Task:** describe exactly one keyframe taken at original t=1.0 s — scene, participants,
  rendered text, platform UI, symbols, composition — and separate what is visible from what is
  inferred. Analytical, not yet political (parties, populism and discourse belong downstream).
- **Input:** the keyframe image plus a bounded context string built by `_row_context()`, which
  caps each labelled field under a character budget.
- **Output:** free-form Markdown, stored in the frame-analysis columns. No schema: the text is
  consumed by Step 4, not parsed.
- **Model settings:** `num_ctx` 8192, `num_predict` 2048, temperature 0.0.
- **Retry:** none. A failed call yields an empty analysis and the quality gate decides
  `REPROCESS` or `DELETE`.
- **Provenance:** `frame_analysis_context_sha256` = SHA-256(`FRAME_PROMPT_VERSION` + context), plus
  the model name. The version is therefore recoverable from the hash only by re-hashing
  candidate versions — see §4.
- **Editability:** the instruction block is research prose. Changing it changes the analysis
  depth and therefore Steps 4–8; bump the version and re-run one FI video before trusting a batch.

### P2 — Step 3, native whole-video analysis

- **Where:** `experiments/vllm_video_test.py`, module constants `SYSTEM_PROMPT` (line ~143, 948
  characters) and `VIDEO_PROMPT` (line ~159, 565 characters), composed into `PROMPT_TEXT`.
- **Version:** `PROMPT_VERSION = "ep24-vllm-video-description-v3-concise"` plus
  `PROMPT_SHA256`, the hash of the composed text.
- **Task:** watch the whole video after the known scroll artifact and describe its narrative in
  temporal order; mark any *additional* feed scroll; end with a quality verdict
  (OK / REPROCESS / DELETE).
- **Input:** the video (native, not stills), plus a metadata context from
  `ep24_metadata_context()`.
- **Output:** `analysis_markdown` plus a trailing JSON object `{SCROLL, SCROLL_SECONDS}`,
  validated against `STRUCTURED_OUTPUT_SCHEMA` (guided decoding where the vLLM build supports
  it; regex extraction and a best-effort regeneration otherwise). A parse failure never discards
  the Markdown.
- **Provenance:** `vllm_video_prompt_version` and `vllm_video_prompt_sha256` columns, plus the
  same pair in the Step 3 SQLite table.
- **Editability:** this is the only stage whose prompt explicitly forbids downstream analysis
  ("no Laclau, no populism, no DNA/SNA"). Removing that sentence would leak interpretation into
  what is supposed to be an evidence representation — treat it as a contract, not style.

### P3 — Step 4, multimodal fusion summary

- **Where:** `roihu_summary.py`, `get_llama_summary_system_prompt()` (8,149 characters) and
  `get_llama_summary_user_prompt()`.
- **Version:** **none.** No `PROMPT_VERSION` constant exists. The prompt is assembled by
  `get_llama_summary_system_prompt()` and `get_llama_summary_user_prompt()`.
- **Task:** fuse the keyframe analysis, the video narrative, ASR/translation and OCR/source
  metadata into one evidence-preserving "Multimodal Social-Semiotic Pre-Analysis", with explicit
  epistemic rules and a fixed output structure.
- **Input:** Step 4's evidence packet, model-visible budget **20,000 characters** (output reserve
  2,000, schema reserve 800; operator override `LACLAUGPT_SUMMARY_EVIDENCE_CHARS`).
- **Output:** free-form Markdown in `summary_analysis`. No JSON schema — Step 5 reads the prose.
- **Retry:** `LACLAUGPT_SUMMARY_EMPTY_RETRIES` default 2 (capped at 3) on empty or truncated
  replies; `num_predict` doubles up to 16,384.
- **Provenance:** `context_sha256` (including the model) is stored in the SQLite cache and the
  Mongo row, so a rerun is reproducible — but there is **no human-readable prompt version**.
  This is the largest provenance gap in the pipeline: Step 4 is the longest prompt and the one
  a reader is most likely to question.
- **Editability:** the 8 KB text mixes research instruction (epistemic rules, what to preserve),
  output-structure boilerplate, and evidence labelling. Any edit must be paired with a new
  version constant (see §5).

### P4 — Step 5, structured post-processing

- **Where:** `roihu_postprocess.py`, `get_system_prompt()` (1,070 characters) and
  `build_postprocess_prompt()`.
- **Version:** **none**, and no prompt hash is persisted.
- **Task:** extract entities, themes and sentiment targets from the Step 4 prose into the
  legacy-compatible surface.
- **Input:** a priority-capped context, `LACLAUGPT_STEP5_MAX_PROMPT_CHARS` default 12,000
  characters: `summary_analysis` 7,800, `asr_transcript` 1,900, then visible text and metadata.
- **Output:** Pydantic `PostprocessResult` — `entities`, `themes`, `positive`, `neutral`,
  `negative`, `video_status` — requested as `format=ResultModel.model_json_schema()`.
- **Retry:** none. An invalid or failed response returns `None` and the row is left for the
  quality gate. This is the only structured stage with no retry loop.
- **Provenance:** model name and settings in the log; **no prompt version and no prompt hash in
  the record**. A row cannot be tied to the exact instruction that produced its entity list.
- **Editability:** the task is deliberately mechanical (extract, do not interpret), which makes
  the prompt safe to tighten — but the missing provenance must be fixed first, or a change
  becomes unattributable.

### P5 — Step 6, Laclau/Palonen discourse analysis

- **Where:** `roihu_populism.py`, `SYSTEM_PROMPT` (line 155, 2,898 characters) plus a retry
  suffix appended on validation failure.
- **Version:** `PROMPT_VERSION = "ep24-laclau-palonen-v3-evidence-packet"`.
- **Task:** operationalise the Laclau/Mouffe/Palonen framework: us-constructs, frontier
  constructs, affect observations, chains of equivalence/difference, nodal/floating/empty
  signifier candidates, rhetorical performances, Palonen dynamics — each with an evidence span,
  confidence, uncertainty notes and counter-evidence.
- **Input:** Step 6's evidence packet, budget **28,000 characters**
  (`LACLAUGPT_STEP6_MAX_CONTEXT_CHARS`); truncation is recorded as `laclau_context_truncated`.
- **Output:** `EP24DiscourseAnalysis` (15 top-level fields; nested evidence classes
  `UsConstruct`, `FrontierConstruct`, `ChainRelation`, `SignifierCandidate`,
  `RhetoricalPerformance`, `PalonenDynamicEvidence`).
- **Retry:** `LACLAUGPT_STEP6_MAX_ATTEMPTS` default 3, with an explicit "ONE COMPLETE
  schema-compliant JSON object" instruction.
- **Provenance:** `laclau_prompt_version` column; a stored row is only resumed when the version
  matches.
- **Editability:** this is the most theory-bearing prompt in the repository. Its definition of
  the populist formula and the evidence rules *are* the operationalisation. Any change is a
  methodological change and belongs in a separate, justified PR — not in a readability pass.

### P6 — Step 7, Discourse Network Analysis

- **Where:** `step_7_roihu_discourse_network_analysis.py`, `SYSTEM_PROMPT` (line 101, 4,376
  characters) plus a `RETRY:` instruction.
- **Version:** `PROMPT_VERSION = "ep24-dna-leifeld-v4-bounded-json"`.
- **Task:** code one statement per actor × concept × stance, Leifeld-style, with an evidence
  quote, the fields it came from, confidence, uncertainty and counter-evidence.
- **Input:** Step 7 evidence packet, budget **20,000 characters**
  (`LACLAUGPT_STEP7_MAX_CONTEXT_CHARS`).
- **Output:** `DNAResult` — `analysis_markdown`, `statements[]` of `DNAStatement` (13 fields:
  actor, canonical actor, concept, canonical concept, proposition, stance, agreement,
  date/time, evidence quote, evidence fields, confidence, uncertainty notes, counter-evidence),
  `corpus_level_cautions`.
- **Retry:** `LACLAUGPT_STEP7_PARSE_ATTEMPTS` default 3; `num_predict` doubles per attempt,
  because truncation is the usual failure.
- **Provenance:** `dna_prompt_version` column; matching version required before a stored row is
  reused.
- **Editability:** the coding rules are the method. The retry text and the JSON reminder are
  boilerplate and may be reworded; nothing else should move without a version bump and a
  re-code of the same sample.

### P7 — Step 8, Social Network Analysis extraction

- **Where:** `step_8_roihu_social_network_analysis.py`, `EXTRACTION_SYSTEM` (line 45, 2,281
  characters).
- **Version:** `PROMPT_VERSION = "ep24-sna-extraction/4.0-textbook-descriptive"`; the graph
  schema carries its own `SNA_SCHEMA_VERSION = "ep24-sna/1.0"`.
- **Task:** extract only explicit, evidenced actor-to-actor relations from the record — not a
  complete network. Deterministic code builds nodes, edges and metrics afterwards.
- **Input:** Step 8 evidence packet, budget **12,000 characters**
  (`LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS`).
- **Output:** `SNAResult` — `analysis_markdown`, `edges[]` of `SNAEdge` (`source_actor`,
  `target_actor`, `relation_type`, `directed`, `evidence_quote`, `confidence`), validated by
  guided schema.
- **Retry:** `LACLAUGPT_STEP8_PARSE_ATTEMPTS` default 3.
- **Provenance:** `sna_prompt_version` and `sna_schema_version` columns. The former
  `sna_castells_prompt_version` column is **retired** and written empty; the Castells
  interpretation is no longer a model call, so there is no second Step 8 prompt to catalogue.
- **Editability:** the "explicit relations only" rule is what keeps Step 8 honest; it is a
  methodological constraint. The descriptive wording around it is not.

### P8 — optional enrichment, entity-resolution adjudication

- **Where:** `ep24_entities.py`, `ollama_adjudicator()` — a fixed system line ("Conservative
  entity-resolution adjudicator. Return JSON only.") and a JSON user payload built from
  `surface_form`, `normalized_form`, `language`, `country`, `candidates` and a one-sentence
  instruction.
- **Version:** **none**, no hash persisted.
- **Task:** choose one supplied `entity_id` only when the match is unambiguous, otherwise return
  an empty id. The prompt explicitly forbids inventing an id.
- **Input:** the candidate list produced by deterministic resolution — the model never sees the
  corpus.
- **Output:** `{"entity_id": "<id or empty>"}`, parsed from a `format="json"` reply at
  temperature 0.0.
- **Retry:** none. A failure leaves the deterministic resolution result in place.
- **Provenance:** none persisted by the prompt itself; the caller records the decision and its
  method in the resolution report (`RESOLVED` / review queue).
- **Editability:** the "never invent an id" constraint is the safety property. Any reword must
  keep an empty answer possible and keep the candidates closed-world.

## 3. Legacy prompts (frozen)

These belong to the Puhti-era implementations that `main` keeps as the compatibility spine.
Their behaviour is pinned by `tests/test_legacy_safety.py` and
`tests/test_legacy_pipeline_contract.py`. **Do not edit them**: the published record depends on
the exact text.

| # | Module | Symbol | Size | Output shape |
|---:|---|---|---:|---|
| 9 | `puhti_frame.py` | prompt inside `get_analysis()` | 1,998 ch | free text, stored in the frame-analysis columns |
| 10 | `puhti_summary.py` | `get_llama_summary_system_prompt()` | 4,465 ch | free Markdown |
| 11 | `puhti_postprocess.py` | `get_system_prompt()` | 2,622 ch | text plus a JSON block |
| 12 | `puhti_populism.py` | module-level `system_prompt` (line 28); user prompt is the row's `summary_analysis` | 10,309 ch | long analytical text plus a machine-readable JSON block |

`puhti_preprocess.py` has no language-model prompt: it runs OCR and ASR only.

## 4. Gaps this catalogue makes visible

1. **Steps 4 and 5 have no prompt version.** They are the second- and third-longest prompts in
   the pipeline. Step 4 records a hash (recoverable, not readable); Step 5 records nothing.
   Recommended fix, one PR: add `PROMPT_VERSION` constants in the same style as Steps 6–8 and
   persist `summary_prompt_version` / `postprocess_prompt_version` additively. No prompt text
   changes in that PR.
2. **Step 5 has no retry loop** while Steps 6–8 do. That is a behaviour difference, not a
   readability issue — record it, do not silently change it.
3. **The version strings carry no date except Step 2's.** `v3`, `v4`, `4.0` are meaningful to
   whoever wrote them; a short `CHANGELOG` line per bump in this document would make the
   sequence auditable. Suggested convention: `ep24-<stage>-v<N>-<short-purpose>-<YYYYMMDD>`.
4. **Prompt text is public, codebook content is private.** The catalogue describes prompt
   structure only; the private codebooks and researcher notes that some prompts carry are
   loaded at runtime and must never be copied here.

## 5. How to change a prompt safely

1. Decide which kind of change it is: **research instruction** (theory, coding rules, evidence
   rules) or **boilerplate** (JSON reminder, retry sentence, output headings).
2. A boilerplate change: edit, run the stage's tests, no version bump required if the rendered
   text is byte-identical after normalisation — otherwise bump.
3. A research-instruction change: bump the version constant, add a line to the change log
   below, re-run the stage on one FI video, and compare the evidence spans — not just the parse
   success. If the change alters classification criteria, it is a methodological change and
   needs its own issue, per #411.
4. Update the row in §1 (size, budget, version) in the same commit; a stale catalogue is worse
   than none.

### Change log

| Date | Stage | Version | Change |
|---|---|---|---|
| 2026-11-16 | — | — | Catalogue created; all versions as found on `main` at `7a03afc`. |
