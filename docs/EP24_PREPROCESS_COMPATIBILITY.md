# EP24 preprocess schema compatibility report (#128 §9)

Machine-checked evidence for what the Roihu preprocess stage reads, what it
writes, and what can safely change. This report exists so that no field is
removed on the strength of its name looking old (#128 §10).

Every claim below is reproducible: the commands are in the appendix.

## 1. What the real incoming data actually is

The canonical EP24 reprocess inputs are
`analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv` in
LaclauGPT-Private. Inspected as materialized Git LFS payloads (not pointers)
with `ep24_input_schema.py`:

```text
country   rows  cols  signature
finland   2424    15   canonical 15-column keep-schema
poland    1862    15   canonical 15-column keep-schema
portugal  1523    23   15 + 8 cleaning fields
bulgaria  1700    23   15 + 8 cleaning fields
croatia   1267    23   15 + 8 cleaning fields
france    2364    23   15 + 8 cleaning fields
germany   1909    23   15 + 8 cleaning fields
hungary   1857    23   15 + 8 cleaning fields
spain     1756    23   15 + 8 cleaning fields
sweden    2591    23   15 + 8 cleaning fields
```

Totals: 10 country files, 19,253 data rows, **2 distinct header signatures**.

### Finding: the "canonical 15-column schema" is the schema for two countries

#128 describes the 15-column layout as "the canonical reprocessing schema".
Measured, it is canonical for **Finland and Poland only**. Eight countries carry
eight additional cleaning fields:

```text
entities_seed_provenance    themes_seed_provenance   researcher_note_source
cleaning_status             cleaning_reason          cleaning_rule_id
include_in_reprocess        needs_human_review
```

Consequence for §7 ("preserve all canonical incoming fields"): a reader that
assumes 15 columns, or that projects each row into the 15-column keep-schema,
**silently drops the cleaning provenance for eight of ten countries**. That is
precisely the lossy projection §7 forbids. The correct behaviour is what
`ep24_schema.source_metadata()` already does — carry every incoming column
dynamically, treating the 15-column list as a contract constant, not a filter.

Required media identity (`video_id`, `allas_filename`) is present in all ten
files, so every country is usable for media stages.

## 2. What this stage writes (field inventory)

`roihu_preprocess.py` appends these columns:

| column | active? | note |
|---|---|---|
| `frame_files` | yes | serialized single-frame path |
| `ocr_1` | yes | the only OCR result ever populated |
| `ocr_2` .. `ocr_6` | **allocated, never populated** | always empty strings |
| `whisperResult` | yes | legacy compatibility alias |
| `whisper_transcript` | yes | ASR text |
| `whisper_language` | yes | detected language |
| `whisper_translated` | yes | English rendering |
| `video_analysis_status` | yes | e.g. `too_short` |
| `video_analysis_note` | yes | human-readable reason |
| `video_initial_skip_seconds` | yes | the 1.0 s scroll boundary |

`ocr_2..ocr_6` are written as `''` unconditionally: `ocr_values = [''] * 6` and
only `ocr_values[0]` is ever assigned. They carry no information today.

## 3. Who reads what downstream (measured)

`ocr_2..ocr_6`:

| consumer | usage | does absence break it? |
|---|---|---|
| `ep24_stage_contract.py` | listed in stage 1 `appends` | **yes** — a test asserts each contracted column is written |
| `tests/test_ep24_stage_contract.py` | `test_every_contracted_append_is_written_by_the_module_it_credits` (passing) | **yes** |
| `roihu_rdf.py` | listed in `PREPROCESS_COLUMNS` | no — the exporter is lossless-by-design and emits unknown columns; it does not index these keys |
| `puhti_*.py` | *historical Puhti* implementations | no — not on the active `main` path |

`whisper_*` / `whisperResult`:

| consumer | refs | note |
|---|---|---|
| `roihu_rdf.py` | 17 | typed predicates + raw legacy emission |
| `roihu_summary.py` | 5 | reads transcript for summarisation |
| `ep24_cleaner.py` | 3 | cleaning decisions |
| `ep24_context.py` | 1 | analysis context assembly |
| `ep24_rag.py` | 1 | RAG corpus |
| `roihu_frame.py` | 1 | frame-stage context |
| `ep24_stage_contract.py` | 1 | stage 1 `appends` |

Tests referencing them: `test_legacy_pipeline_contract.py` (6),
`test_ep24_cleaning.py` (6), `test_roihu_rdf.py` (6),
`test_ep24_canonical_pipeline.py` (3), `test_ep24_video_scroll.py` (2),
`test_asr_backend.py` (1).

## 4. Conclusion: what is safe, and what is not

**Safe without further permission (additive only):**

- reading the real 15/23-column inputs instead of a guessed legacy projection;
- the input-schema inspector and the single-frame regression tests;
- adding generic ASR output fields *alongside* the existing `whisper_*` fields.

**Not safe without an explicit decision — three concrete blockers:**

1. **Removing `ocr_2..ocr_6` fails a passing test today.**
   `ep24_stage_contract.py` contracts stage 1 to append them, and
   `test_every_contracted_append_is_written_by_the_module_it_credits` passes.
   Any removal must update the contract in the same commit. That is a
   *deliberate contract change*, not cleanup, and it is exactly the kind of
   schema removal `AGENTS.md` reserves for explicit human permission:

   > Agents "must not remove, replace, collapse, silently rewrite, or make a
   > legacy step unavailable **without explicit human permission**" and
   > "**Do not perform opportunistic destructive refactors, schema removals**".

2. **Renaming `whisper_*` to `asr_*` breaks seven live consumers.** §3 asks for
   backend-neutral names. With `roihu_summary`, `ep24_context`, `ep24_rag`,
   `ep24_cleaner`, `roihu_rdf`, `roihu_frame` and the stage contract all reading
   `whisper_*`, the rename needs either a compatibility alias written alongside
   the new fields (recommended: additive, matches the repo rule) or a
   coordinated migration across all seven plus their tests. §10 says "old
   `whisper_*` schema once migrated" — the migration is not done, so the
   precondition is not met yet.

3. **The legacy branch is immutable.** `AGENTS.md`: never modify, merge into,
   rebase or backport to `legacy`; the historical CSV semantics must stay valid
   for published research. Removing columns from the active path is fine *in
   principle* as long as `legacy` and the historical field meanings are
   untouched — but that is a further reason to add new fields beside the old
   ones rather than in place of them.

### Recommended path (matches the repo's own rule)

Add `asr_transcript` / `asr_language` / `asr_translated` / `asr_backend` /
`asr_model` **as new columns**, and keep `whisper_*` populated for as long as any
consumer reads them. Then, once consumers are migrated and a test proves nothing
reads the old names, retire `whisper_*` and `ocr_2..ocr_6` in one commit that
also updates `ep24_stage_contract.py` and its test.

## 5. What §3 and §4 need that this environment cannot supply

§3 (modern ASR benchmark) and §4 (modern OCR benchmark) require GPU runs on CSC
Roihu with real EP24 clips from at least Finland, Poland and Portugal. That needs
Tomi's CSC allocation and credentials, and Laskin GPU work is out of scope by
standing instruction. What is deliverable here is the **evaluation protocol**
(candidate matrix, clip selection, metrics, reproducibility requirements) so the
benchmark is runnable on Roihu — not a benchmark result. No benchmark numbers are
claimed in this report.

## Appendix: reproducing these numbers

```bash
# incoming schema, all ten countries (read-only; prints names/counts/hashes only)
python3 ep24_input_schema.py --root /path/to/LaclauGPT-Private
python3 ep24_input_schema.py --root /path/to/LaclauGPT-Private --json

# downstream consumers
grep -rn "ocr_[2-6]" --include=*.py .
grep -rn "whisper_" --include=*.py .

# the test that gates ocr_2..ocr_6 removal
python3 -m pytest tests/test_ep24_stage_contract.py -q
```

The inspector refuses LFS pointer stubs by design, so running it against an
unsmudged checkout fails loudly instead of reporting zero rows as if the data
were empty.
