# Researcher's guide to the EP24 analysis pipeline

This guide is the **reading map** for researchers who need to understand, reproduce, or challenge an analytical result. It does not change analysis behavior, model instructions, or storage contracts. For the machine-readable specification see [the stage contract](../ep24_stage_contract.py), [the cumulative CSV contract](EP24_PIPELINE_CONTRACT.md), and [the operating guide](ROIHU_NUMBERED_PIPELINE.md).

## Read the pipeline in five minutes

**One row represents a source item** (usually one video). Each stage receives a cumulative table and adds fields without dropping earlier fields. A research claim in a late stage should be traceable back to the source item and earlier evidence. The model is an *interpreter of supplied evidence*, not a source of new observations.

| Step | Research question | Main code to read | How the result is produced |
| --- | --- | --- | --- |
| 0. Import/bootstrap | Which source item and researcher annotations are we analysing? | `step_0_roihu_import.py`, `ep24_bootstrap.py`, `ep24_schema.py` | Deterministic identity and schema preparation |
| 1. Preprocess | What speech and visible text are present? | `step_1_roihu_preprocess.py` → `roihu_preprocess.py` | Media extraction, OCR and ASR/translation; backend/model provenance recorded |
| 2. Frame | What is visible in the selected still image? | `step_2_roihu_frame.py` → `roihu_frame.py` | Model-assisted description of the single source-time 1.0s frame |
| 3. Video | What happens over time in the clip? | `step_3_roihu_video.py` → `experiments/vllm_video_test.py` | Optional whole-video VLM; scroll quality controls |
| 4. Summary | What combined account is supported by the different modalities? | `step_4_roihu_summary.py` → `roihu_summary.py` | Evidence-preserving fusion of frame, video, OCR, ASR and metadata |
| 5. Postprocess | Which entities, themes and sentiment targets can be structured? | `step_5_roihu_postprocess.py` → `roihu_postprocess.py` | Legacy-compatible normalization and structured extraction |
| 6. Discourse | How is political meaning articulated? | `step_6_roihu_discourse_analysis.py` → `roihu_populism.py` | Laclau/Palonen-informed interpretation; **not** an observed social tie |
| 7. DNA | Who makes which supported or opposed claim? | `step_7_roihu_discourse_network_analysis.py`, `ep24_dna.py` | Model-assisted actor–concept statement coding with provenance |
| 8. SNA | Which social/communication relations are evidenced? | `step_8_roihu_social_network_analysis.py`, `ep24_sna.py` | Model-assisted extraction of evidenced events, then deterministic graphs and metrics |
| 9. RDF | How can the resulting records and graph be queried? | `step_9_roihu_rdf.py` → `roihu_csv_rdf.py` | Deterministic export of **existing** findings; not a new interpretation |

The numbered `step_*.py` files are the public entry points, **not duplicates to delete**. Several deliberately forward into historical `roihu_*.py` implementation files. The `legacy` branch is an immutable published research record and must not be edited.

## Where to find things

| Location | What belongs here |
| --- | --- |
| `step_N_roihu_*.py` | Direct, numbered CLI entry points; one per stage |
| `roihu_*.py` | Existing compatibility stage implementations; do not remove because a wrapper exists |
| `ep24_*.py` | Shared contracts, configuration, provenance, schema, storage, graph, and orchestration logic |
| `scripts/roihu/` | Slurm jobs, setup/submit wrappers, and operational tooling |
| `scripts/ep24/` | Import, reporting, QA, export and rollout utilities |
| `config/` | Public schema and example configuration; **no secrets** |
| `docs/` | Research/operational contracts, methodological cautions and historical audit reports |
| `tests/` | Synthetic fixtures and regression tests; no confidential research records |

**Start here:** `README.md` (project orientation), this guide (concepts and evidence), `docs/ROIHU_NUMBERED_PIPELINE.md` (operation), `docs/EP24_PIPELINE_CONTRACT.md` (data meanings), `docs/CODING_STYLE.md` (readable implementation conventions), `AGENTS.md` (non-negotiable protections).

## Following one analytical claim

1. Identify the original source through `video_id`, `allas_filename`, `country`, and `_storage_id` (where present). IDs must not be regenerated merely because a stage is rerun.
2. Inspect the cumulative CSV: earlier columns survive as later stages append fields. Compare the relevant row in the previous and current checkpoint rather than reading a final narrative alone.
3. Inspect input evidence (source video, frame, OCR, transcript and model-visible source packet). The initial feed-scroll artifact affects visual analysis; see [video handling](EP24_VIDEO_HANDLING.md). Step 1 ASR intentionally analyses the full video.
4. Read the stage's system prompt and input builder **as executed by that stage**, not a prose paraphrase of what the prompt should say. Stage-specific prompt constants live in source; `ep24_prompt_contract.py` defines shared evidence/context packet rules and fingerprints for applicable stages.
5. Check the response validation, normalization, evidence quotation, missing-value handling and retry path. A plausible model claim unsupported by source material is not a valid observation.
6. Inspect the appended structured and Markdown fields, stage status, model/runtime identification and prompt/version provenance. Missing results and failed validations must not be mistaken for a coded absence.
7. For DNA, keep actor–concept stance separate from SNA actor–actor ties. For SNA, distinguish model-extracted relationship events from deterministically calculated graph measures; network interpretations cannot create evidence or alter graph topology.
8. When uncertain, compare with the applicable schema/tests and record the limitation rather than inferring absent evidence.

### A conceptual (synthetic) example

Suppose a source video contains a verifiable quote, “Party A opposes proposal B.” This is **illustrative, not a real research observation**. Step 1 may transcribe those words; Step 4 may summarize the clip; Step 6 may interpret its discourse; Step 7 may encode an actor–concept statement (`Party A` / `proposal B` / oppose), *if* the quote and actor attribution meet its rules. The statement does **not** automatically establish a social interaction between Party A and some other actor. Step 8 should not fabricate an actor–actor edge. Step 9 may serialize only what earlier stages actually produced.

## How prompts should be read

A prompt has four distinct layers: (1) **research objective**, (2) **source evidence provided in this run**, (3) **researcher-defined coding rules and codebook/context**, and (4) **required output schema and validation conditions**. Do not collapse these layers.

- Shared model-visible evidence-versus-context distinctions and token/character budget behavior: [`ep24_prompt_contract.py`](../ep24_prompt_contract.py) and [prompt runbook](EP24_PROMPT_EVIDENCE_RUNBOOK.md).
- Step 6 interpretation prompt: [`roihu_populism.py`](../roihu_populism.py).
- Step 7 DNA system prompt: [`step_7_roihu_discourse_network_analysis.py`](../step_7_roihu_discourse_network_analysis.py), constant `SYSTEM_PROMPT`.
- Step 8 SNA extraction prompt: [`step_8_roihu_social_network_analysis.py`](../step_8_roihu_social_network_analysis.py), constant `EXTRACTION_SYSTEM`.
- For other stages, follow the named entry-point arrows above to the actual implementation and search for the prompt text passed to its backend. A prompt constant is not necessarily the complete final model-visible packet.

Research code's detailed prompts may be long for good methodological reasons. Any rewriting that changes substantive criteria, priorities, examples, or JSON requirements is a *methodological modification*, not harmless code cleanup. Real restricted prompts and codebooks must stay in private storage.

## Operational reality and verification

Follow [the Roihu operating guide](ROIHU_NUMBERED_PIPELINE.md) for current submission commands, checkpoint/resume semantics and Slurm logs; use [storage documentation](MONGODB_STORAGE.md) to distinguish MongoDB durable state, Redis coordination, Allas media and cumulative CSV. Do not assume that one local Python command can reproduce a Roihu GPU job without its dependencies, configured private material, and source data.

For checks that do not require real media or private infrastructure, run focused regression tests, such as:

```bash
python -m pytest tests/test_ep24_stage_contract.py tests/test_prompt_contract.py
```

Before declaring a refactor behavior-preserving, compare stage schemas, prompt fingerprints, representative synthetic results, CLI defaults, stdout/log behavior, and checkpoint/resume paths. LLM responses may vary; deterministic processing can be checked by exact equality, while model-inference comparisons need bounded, evidence-based assessment.

## Glossary

- **ASR:** Automatic speech recognition, turning audio into a transcript.
- **OCR:** Optical character recognition, reading text in images.
- **VLM:** Vision-language model, analysing images/video and text.
- **Cumulative CSV:** A table that preserves earlier columns and appends later-stage results.
- **Provenance:** Evidence of where a value came from, which model/prompt processed it and how it was transformed.
- **Codebook:** Researcher-defined analytical categories and operational coding definitions.
- **DA / discourse analysis:** Interpretation of the articulation of political meaning; here mainly Laclau/Palonen-informed.
- **DNA / Discourse Network Analysis:** Statements connecting an actor, concept, stance, source and time; not automatically communication ties.
- **SNA / Social Network Analysis:** Explicitly supported actor-to-actor connections and descriptive graph metrics.
- **RDF:** Resource Description Framework, a structured representation used for linked/queryable graph data.
- **Checkpoint:** Durable intermediate state used for inspection or safe resumption.
- **Prompt fingerprint:** Reproducibility metadata identifying the effective model-visible instructions/configuration; see its concrete implementation for scope.

## Safe future simplification (issue #411)

The current tree has multiple operational wrappers and historically accumulated reports. **Similarity of names is not proof of redundant behavior.** Audit invocation references, field contracts, Slurm scripts, tests, publication dependencies and private workflows before moving or consolidating files. Keep numbered entry points and compatibility implementations until a tested migration is explicitly approved.

The biggest readability gains are likely to come from reducing unclear navigation, improving module docstrings where thin, and maintaining a **single discoverable guide** rather than rewriting established human research logic. This document provides that starting point.
