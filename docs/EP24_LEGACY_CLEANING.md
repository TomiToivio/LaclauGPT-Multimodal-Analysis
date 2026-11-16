# EP24 legacy cleaning and reprocessing input contract

> **Issue #21 canonical handoff:** this document also describes the historical
> pre-migration source fields used by legacy cleaning. Those four fields
> (`new_entity`, `researcher_new_persons`, `new_theme`,
> `researcher_new_themes`) may be read only while converting historical
> artifacts. They must not cross the reprocessing boundary. Current Step 1 input
> contains only canonical human `entities` and `themes` JSON lists, and later
> model stages must preserve those fields unchanged.


Issue #35 defines a non-destructive bridge from the historical EP24 country CSVs to the
new Roihu/Phase 2 reprocessing pipeline.

Two cleaners are **maintained on purpose**, and they answer different questions (issue #413, approved
option **C**). Neither is being retired; a reader should pick by input, not by seniority.

| | `ep24_cleaner.py` (526 lines) | `roihu_clean.py` (1,100 lines) |
| --- | --- | --- |
| **Job** | Clean a per-country CSV **against the shared keep-schema workbook** | Clean a per-country CSV **against the schema built into the module** (`KEEP_SCHEMA_ORDER`) |
| **Input** | `--input <csv> --keep-schema <xlsx> --output-dir <dir> --country <name>` | `--source <csv> --to-reprocess-dir <dir> [--country] [--output-dir]` |
| **Schema source** | The authoritative workbook, read at run time, so the schema can be revised without touching code | A module constant, so a run is reproducible from the code alone |
| **Report** | Markdown cleaning report plus the reprocessing outputs | `CleaningOutcome`/`RowDecision` detail, duplicate summary and a build report |
| **Extra machinery** | Recut worklist, human-label merging, researcher-note normalisation | Tri-state parsing (`TriState`), obsolete-column resolution, seed-value merging, multi-value splitting |
| **Used by** | `tests/test_ep24_cleaner.py`, and documented as the operator path below | `tests/test_ep24_cleaning.py`, and the per-country audits in `docs/ep24_country_audits/` whose numbers it produced |
| **Choose it when** | The keep-schema workbook is the thing you want to vary — a new country, a revised column set | You need the audit-compatible path that reproduces published per-country results |

**Rule of thumb:** `ep24_cleaner` is the schema-driven path for current work; `roihu_clean` is the
code-driven path for reproducing audit figures. They share function names, not a single job, and the
overlap is deliberate rather than an accident of history.

`ep24_legacy_clean.py` (646 lines), a third unreferenced implementation, was **removed from `main` in
#413** and is recoverable from Git history and the immutable `legacy` branch where applicable. Real EP24
data, researcher notes, country codebooks and generated derivatives remain in `LaclauGPT-Private`.

## One schema for every country

The workbook

`LaclauGPT-Private/analysis/ep24_reprocess/research_keep/ep24_finland_keep_columns.xlsx`

is the **single authoritative keep-schema for all EP24 countries**. Despite its filename,
it is not Finland-specific. Finland and Poland are merely the first demo countries.

Country differences belong in bilingual codebooks, alias dictionaries, political actors,
terminology, and local research context. They do not create different dataframe schemas.

## Input and output rules

The source country CSV is historical provenance and must never be overwritten.

For each country the cleaner creates derivative artifacts:

- `data/to_reprocess/ep24_<country>_cleaned.csv`
- `ep24_<country>_cleaning_decisions.csv`
- `ep24_<country>_cleaning_report.md`
- `ep24_<country>_cleaning_provenance.json`

Cut/split cases may additionally be appended to the shared private worklist
`analysis/ep24_reprocess/data/ep24_videos_to_recut.csv`.

The reprocessing CSV is synthetic-profile only. `account_type == Synthetic` is the
authoritative current filter. Organic rows remain untouched in the historical source.

Rows are excluded when either `researcher_delete_video` or
`researcher_dubious_video` is explicitly true. Cut/split flags route the row to the
recut worklist instead of normal reprocessing. Blank researcher fields are unknown or
unreviewed and are never converted to false.

`mobilebackup` is rejected as a pseudo-country. Country identity must come from
authoritative metadata, not a storage path component.

## Human supervision carried forward

For historical source files, the cleaner reconstructs recyclable human seed
fields before the canonical handoff:

- `entities` = verified/canonicalized union of historical entity/person fields;
- `themes` = verified/canonicalized union of historical theme fields.

The reprocessing CSV then drops the four historical annotation columns. From
Step 1 onward, `entities` and `themes` are immutable human annotations;
machine-extracted entities/themes are stored separately.

Aliases should be supplied by the relevant bilingual country codebook. Deduplication is
performed after canonicalization. Unresolved plausible labels can remain provisional and
their raw values/source columns remain in provenance JSON.

These reconstructed `entities` and `themes` are human-informed seed metadata. They
must not be confused with the legacy model-generated `entities`, `topics`, or other
old analysis outputs.

## Researcher notes

Researcher-note triage happens only after delete/dubious and cut/split filtering.

- blank means no explicit researcher comment;
- `Test` is normalized away as a trivial note;
- the explicit unusable example `This is filthy garbage` is excluded;
- other non-trivial notes are retained as `human_researcher` context and marked for
  human review rather than silently converted into model ground truth.

## Old analysis outputs

Legacy machine outputs are diagnostic material only. They should be studied separately
for recurring hallucinations, omissions, alias fragmentation, translation problems,
sentiment-target errors, OCR/frame grounding failures, and prompt pathologies.

The reprocessing input does not recycle old LDA, summary, populism, topic, sentiment,
Whisper, OCR, or frame analysis as authoritative metadata. All `lda_*` fields are
obsolete. OCR/frame/Whisper/video analysis is regenerated.

Spreadsheet-position removals from issue #35 are resolved against the actual header
before removal: Y, Z, AA, AB, AE, and AH through AT. The report records the resolved
field names.

## Media rule

Every newly derived media analysis starts at **t = 1.0 seconds**. This includes whole
video analysis, Whisper, frame extraction, and frame-derived OCR. The source video is
never rewritten. Later scroll events should be flagged for recut/split review.

## Command-line example

Run from the public repository while pointing at private files:

```bash
python ep24_cleaner.py \
  --input ../LaclauGPT-Private/analysis/ep24_reprocess/data/by_country/ep24_finland_with_researcher_notes.csv \
  --keep-schema ../LaclauGPT-Private/analysis/ep24_reprocess/research_keep/ep24_finland_keep_columns.xlsx \
  --output-dir ../LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess \
  --recut-worklist ../LaclauGPT-Private/analysis/ep24_reprocess/data/ep24_videos_to_recut.csv \
  --country Finland
```

Use the same keep-schema argument for Poland, Germany, and every other EP24 country. The canonical cleaned derivative filename is `ep24_<country>_cleaned.csv` for every country.

Public tests use synthetic rows only. Never copy real researcher notes or private EP24
rows into this repository.
