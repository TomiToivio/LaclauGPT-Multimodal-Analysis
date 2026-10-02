# EP24 Step 1 schema compatibility and Roihu benchmark gate

Issue #128 migrates the active EP24 reprocessing path away from the historical six-frame / Whisper-named schema.

## Authoritative input

The materialized private `to_reprocess` files were inspected on 2026-10-02. There are **10 files, 19,253 rows, and two real header signatures**:

| country | rows | columns |
|---|---:|---:|
| Finland | 2,424 | 15 |
| Poland | 1,862 | 15 |
| Portugal | 1,523 | 23 |
| Germany | 1,909 | 23 |
| Spain | 1,756 | 23 |
| Hungary | 1,857 | 23 |
| Croatia | 1,267 | 23 |
| France | 2,364 | 23 |
| Bulgaria | 1,700 | 23 |
| Sweden | 2,591 | 23 |

The 15-column identity/keep schema is:
`video_id, country, author_username, account_type, source_type, source_recording, sequence_number, political_preference, allas_filename, new_entity, new_theme, video_duration, researcher_new_persons, researcher_new_themes, researcher_note`.

Eight countries additionally carry:
`entities_seed_provenance, themes_seed_provenance, researcher_note_source, cleaning_status, cleaning_reason, cleaning_rule_id, include_in_reprocess, needs_human_review`.

Those eight fields are real incoming provenance and **must be preserved**. The active reader is therefore deliberately additive/dynamic rather than a projection to 15 columns. Required media identity (`video_id`, `allas_filename`) is present in all ten materialized files.

GitHub's ordinary file view still exposes LFS pointer stubs rather than private rows. Step 1 therefore refuses pointer files and logs SHA256, row count, ordered columns, dtypes, and missing media fields after materialization. No private rows are copied into public fixtures or documentation.

## Active Step-1 output

Every incoming field is preserved, then Step 1 appends:

- `frame_file` and `frame_timestamp_seconds` (exactly one original-video still at 1.0 s)
- `ocr_1`, `ocr_backend`, `ocr_model`, `ocr_runtime_ms`
- `asr_transcript`, `asr_language`, `asr_translated`, `asr_backend`, `asr_model`, `asr_runtime_ms`
- `video_duration_seconds`
- `preprocess_status`, `preprocess_note`, `preprocess_completed_at`

The active writer does not create `ocr_2..ocr_6`, `frame_files`, `whisperResult`, or `whisper_*`.

Frozen historical artifacts can still be read downstream through temporary read-only transcript fallbacks. They are not regenerated.

## Storage / recovery

MongoDB remains the canonical shared durable record under `ep24_stage_orchestrator.py`. SQLite `database/preprocess_v2.db` is a job-local restart cache only. CSV remains a first-class cumulative interchange artifact. A successful direct Step-1 CSV write also creates a timestamped backup; the orchestrator additionally checkpoints cumulative CSV + SQLite state after every durable MongoDB batch.

## Country order

The direct Step-1 order is pinned to:

1. Finland
2. Poland
3. Portugal
4. Germany
5. Spain
6. Hungary
7. Croatia
8. France
9. Bulgaria
10. Sweden

## ASR decision and benchmark gate

The dataframe contract is backend-neutral. The candidate adapters are:

- NVIDIA Canary 1B v2 (default candidate): 25 European languages including all EP24 languages; direct X→English translation.
- NVIDIA Parakeet-TDT 0.6B v3: 25 European languages, automatic language identification, punctuation/capitalization and timestamps.
- Qwen3-ASR 1.7B: strong noisy/BGM use case and automatic language ID, but its published language list does not include Bulgarian or Croatian, so it cannot be the sole all-country backend.
- Whisper / faster-whisper: comparison baseline only.

The final production selection must be measured on materialized restricted EP24 clips on Roihu GH200, at minimum Finland, Poland and Portugal, recording accuracy/researcher correction, language ID, punctuation, runtime and VRAM. Do not claim that benchmark is complete from public CI.

The private benchmark manifest must now include `reference_transcript_provenance` and `reference_ocr_provenance`. Any non-empty reference must be explicitly human/researcher/manual/gold verified. Earlier EP24 transcript annotations that are themselves Whisper output are useful historical artifacts but **must not be used as ASR ground truth**, because that would reward candidates for reproducing Whisper rather than the speech.

## OCR decision and benchmark gate

PaddleOCR 3.x is the modern default candidate, configured to PP-OCRv5 for broad multilingual coverage. EasyOCR remains selectable as the historical comparison. OCR runs once on `frame_file` only.

The final production selection must compare both backends on the same real t=1.0 s EP24 frames on Roihu and record recognition accuracy plus runtime. Public synthetic tests verify call count and schema, not private-data accuracy.

## Downstream audit

Current active consumers were migrated as follows:

- Step 2 reads `frame_file`.
- Step 4 summary prefers `asr_translated` / `asr_transcript` and consumes only `ocr_1`; old transcript names are read-only fallback.
- RAG/context retrieval prefers generic ASR and keeps legacy fallback for old persisted rows.
- `ep24_stage_contract.py` and `config/ep24_pipeline_columns.json` define the new Step-1 additions.

Historical `puhti_*.py`, the frozen legacy contract document, and legacy fixtures intentionally remain historical evidence and are not the active reprocess schema.

## Completion boundary

Code/schema migration and the **all-ten materialized input-schema audit are complete**. The issue should remain open only for the acceptance checks that require real media on CSC Roihu:
- Finland → Poland → Portugal Step-1 smoke results in the required order;
- real-clip ASR benchmark on representative Finland/Poland/Portugal EP24 media, including accuracy/corrections, language ID, punctuation, runtime and VRAM;
- real-frame OCR benchmark comparing PaddleOCR and EasyOCR on the same t=1.0s frames;
- the resulting production ASR/OCR backend selections and reproducible Roihu install notes.

The RDF exporter has an explicit regression test for the active `asr_*` producer shape as well as the frozen Whisper fallback, so semantic transcript nodes cannot silently disappear while legacy-only RDF fixtures remain green.
