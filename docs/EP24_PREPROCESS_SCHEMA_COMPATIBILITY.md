# EP24 Step 1 schema compatibility and Roihu benchmark gate

Issue #128 migrates the active EP24 reprocessing path away from the historical six-frame / Whisper-named schema.

## Authoritative input

The private Finland cleaning report records the canonical 15-column Step-0 reprocess schema:

`video_id, country, author_username, account_type, source_type, source_recording, sequence_number, political_preference, allas_filename, new_entity, new_theme, video_duration, researcher_new_persons, researcher_new_themes, researcher_note`.

The public GitHub view of all ten private CSVs is a Git LFS pointer, not the materialized dataset. Step 1 therefore refuses LFS pointer files and logs SHA256, row count, ordered columns, dtypes, and missing media fields from the materialized file before processing. No public fixture contains private EP24 rows.

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

Code/schema migration can be reviewed in normal CI. The issue itself should remain open until a Roihu run records:
- all ten materialized CSV headers;
- Finland → Poland → Portugal smoke results;
- real-clip ASR benchmark;
- real-frame OCR benchmark.
