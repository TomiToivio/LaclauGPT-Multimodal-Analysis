# Legacy EP24 multimodal pipeline: stage and schema contract

This document is the **authoritative written contract** for the historical EP24
multimodal pipeline, reconstructed by reading the frozen scripts on the
[`legacy`](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/tree/legacy)
branch.

It exists because the Roihu migration has to preserve a sequence of
**file- and column-level agreements between five scripts** that were previously
only expressed in the code itself. `docs/MIGRATION_ROIHU_STAGE0.md` inventories
paths, models and platform assumptions; this document covers what that one left
out: **what each stage reads, what it writes, which columns it creates, how it
fails, and what the final dataframe must contain.**

Status: documentation only. Nothing here changes pipeline behaviour.

Source of truth: the code on `legacy`. Where this document and the code disagree,
**the code is right and this document is a bug.**

---

## 1. Stage order

Five scripts run in this order. The order is load-bearing: each stage consumes
the previous stage's CSV and rewrites it in place.

| # | Legacy script | Reads | Writes | Durable state |
|---|---|---|---|---|
| 1 | `puhti_preprocess.py` | `./csv/tiktok_videos.csv` | `./csv/tiktok_<language>.csv` | `./database/preprocess.db` |
| 2 | `puhti_frame.py` | `./csv/tiktok_<language>.csv` | `./csv/tiktok_<language>.csv` | `./database/frame.db` |
| 3 | `puhti_summary.py` | `./csv/tiktok_<language>.csv` | `./csv/tiktok_<language>.csv` | `./database/summary.db` |
| 4 | `puhti_postprocess.py` | `./csv/tiktok_<language>.csv` (fallback `./ep24_<language>.csv`) | `./csv/tiktok_<language>.csv` **and** `./ep24_<language>.csv` | none |
| 5 | `puhti_populism.py` | `./ep24_<country>.csv` | `./ep24_<country>.csv` | `./formula_of_populism.db` |

All paths are relative to the **current working directory**. The Roihu runner
therefore executes the public scripts with the private runtime root as CWD rather
than rewriting the paths (see `docs/ROIHU_MIGRATION.md`).

The `ep24_<language>.csv` file written by stage 4 is the legacy compatibility
artifact and is the **only** input stage 5 accepts.

### The target names on the Roihu branch

The issue specifies `roihu_*.py` entry points. Those names are the goal for the
new stages; the `legacy` scripts above are the behavioural reference they must
reproduce. Do not assume a `roihu_*` script exists until it has been added.

---

## 2. Entry-point behaviour (module level, not `main()`)

Each script is written as a **module that runs on import**. There is no
`main()`, no `argparse`, and no `if __name__ == '__main__'` guard. Consequences
that the migration must reproduce or deliberately decide about:

- **Import-time side effects.** `puhti_preprocess.py` calls `os.makedirs('./logs')`
  and `os.makedirs('./database')` at module top level; four of the five scripts
  open a SQLite connection at import. Importing a module therefore writes to the
  CWD.
- **The whole run is one process.** Each script ends with a module-level loop
  over the full language/country list, so a single invocation processes every
  language in sequence.
- **Heavy objects are constructed once.** `puhti_preprocess.py` builds the
  EasyOCR reader and loads the Whisper `large` model at import, before any row is
  read.

### Language and country lists

| Stage | List |
|---|---|
| `puhti_preprocess.py` | `['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']` |
| `puhti_frame.py` | `['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']` |
| `puhti_summary.py` | `['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']` |
| `puhti_postprocess.py` | `['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']` |
| `puhti_populism.py` | `['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg']` |

Two asymmetries are historical and must **not** be silently normalised:

1. **Stages 1–4 iterate both `bg` and `en`; stage 5 iterates `bg` only.** `en` has no `ep24_en.csv`
   produced by stage 5.
2. **The EasyOCR reader list is not the stage language list.** Preprocess builds
   `easyocr.Reader(['en', 'fr', 'pl', 'sv', 'pt', 'de', 'es', 'hu', 'hr'])` — nine
   languages, and `fi` is absent from that list while present in the stage list.
   This is the historical OCR configuration; treat it as a measured constraint to
   preserve, not as a typo to fix.

---

## 3. Per-stage contract

### Stage 1 — `puhti_preprocess.py` (preprocessing)

**Reads** `./csv/tiktok_videos.csv`. **Writes** `./csv/tiktok_<language>.csv`.
**State** `./database/preprocess.db`, table `tiktok_videos`.

Input columns the stage requires: `language`, `authorUniqueId`, `videoId`,
`scrapedCountry`, and optionally `whisperResult`.

Behaviour:

- Ensures a `whisperResult` column exists, **adding it empty if absent**. Rows are
  never dropped on `whisperResult` here — this stage is what creates the
  transcript.
- Resets these columns to `''` before the row loop: `frame_files`, `ocr_1` …
  `ocr_6`, `whisper_transcript`, `whisper_language`, `whisper_translated`.
- Filters to `language == <the current loop value>`.

Per row, keyed by `(author_username, video_id)`:

1. Look up `./database/preprocess.db`. If a cached row exists, take all cached
   values from the database and **continue** — no media is touched.
2. Otherwise resolve the video at
   `./Allas/Scraper/TikTok/Videos/<scrapedCountry>/<authorUniqueId>/<videoId>.mp4`.
   If it does not exist, log an error and **continue** (row is left with empty
   values and still written to the CSV).
3. Extract up to six keyframes, one every 30 seconds over
   `max(1, min(int(duration), 180))` seconds, saved under
   `./Keyframes/TikTok/<author_username>/<video_id>/<n>.jpg` (`n` from 1).
4. Run EasyOCR on each frame; frame *i* becomes `ocr_<i>`, with results joined by
   newlines.
5. Transcribe with Whisper `large` (`download_root='./whisper/'`), temperature
   ladder `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. When the detected language is not
   English, translate the first 3000 characters to English with
   `deep_translator.GoogleTranslator`; when it is English, the translation is the
   transcript itself.
6. Write the cache row, then set every output column on the dataframe.

Row failures are caught and logged (`logger.exception`); the run continues.

**Field the stage sets that later stages depend on:**

- `whisperResult` — the resolved best transcript. Computed by `best_transcript()`,
  which returns the **first non-empty, non-`'nan'`** value in the order
  `(legacy whisperResult, whisper_transcript, whisper_translated)`. This
  deliberately preserves pre-existing legacy transcript data rather than
  overwriting it.
- `frame_files` — comma-separated keyframe paths. Serialised by
  `normalize_frame_files()`, which accepts both a `str(list)` (older cache rows)
  and a comma-separated string, so old preprocess databases keep working
  downstream.

### Stage 2 — `puhti_frame.py` (frame / multimodal analysis)

**Reads and writes** `./csv/tiktok_<language>.csv`. **State**
`./database/frame.db`, table `tiktok_videos`.

- `dropna(subset=['whisperResult'])` then `dropna(subset=['frame_files'])` —
  **this stage is the first to drop rows.**
- Creates `frame_analysis_1` … `frame_analysis_6`, initialised to `''`.
- Cache key is `(author_username, video_id)` in `frame.db`.

Per row: split `frame_files` on `,`; for each frame call the vision model with a
**descriptive social-semiotic pre-analysis** prompt and wrap the response as:

```
### **Frame <n> at <seconds> seconds**:
<model response>
```

`seconds` is `i * 30` for frame index `i` (0-based). The wrapped text — heading
included — is what is stored, not the raw model output.

The prompt is explicitly **pre-discursive**: it forbids political, ideological,
populism, sentiment and Laclauian analysis, which is reserved for stage 5.

### Stage 3 — `puhti_summary.py` (summary)

**Reads and writes** `./csv/tiktok_<language>.csv`. **State**
`./database/summary.db`, table `tiktok_videos`.

- `dropna(subset=['whisperResult'])`.
- Creates `summary_analysis` and `metadata`.
- Requires `authorNickname`, `authorSignature`, `authorUniqueId`, `videoId`,
  `videoCreated` (epoch seconds), `videoDuration`, `videoDiggCount`,
  `videoShareCount`, `videoCommentCount`, `videoPlayCount`, `videoDescription`,
  plus the `frame_analysis_*` and `ocr_*` columns from earlier stages.
- Cache key is `(author_username, video_id)` in `summary.db`.

`metadata` is a **fixed-format text block** assembled from the row, and the video
URL and author URL are derived as
`https://www.tiktok.com/@<authorUniqueId>/video/<videoId>` and
`https://www.tiktok.com/@<authorUniqueId>`. Hashtags are extracted from
`videoDescription` by splitting on whitespace and keeping tokens starting with
`#`, joined with `, `.

The frame analyses and their matching OCR text are concatenated into one
`frame_analysis` payload before the call, with OCR presented under
`### OCR results for frame <n> at <seconds> seconds:`. **The `seconds` values
are hard-coded (0/30/60/90/120/150), not recomputed from the media.**

The summary prompt is again descriptive and explicitly forbids discourse,
political and Laclauian analysis.

### Stage 4 — `puhti_postprocess.py` (structured extraction / compatibility output)

**Reads** `./csv/tiktok_<language>.csv`, falling back to `./ep24_<language>.csv`
with a logged warning. **Writes both** `./csv/tiktok_<language>.csv` and
`./ep24_<language>.csv`. No SQLite state.

- Requires `summary_analysis`; if absent it logs an error and **skips the whole
  language**.
- Creates `entities`, `topics`, `positive`, `neutral`, `negative`.
- Each row's `summary_analysis` is sent to the model with a JSON schema
  (`Sentiment`), and the five lists are written as `, `-joined strings with
  exact duplicates removed.
- Rows with a missing or blank `summary_analysis` are skipped individually.

`ensure_video_filename()` creates the **join key** that stage 5 depends on:

- `authorUniqueId/videoId` when both columns exist;
- otherwise `videoId`;
- otherwise the dataframe row index, with a warning.

### Stage 5 — `puhti_populism.py` (Laclau / Palonen analysis)

**Reads and writes** `./ep24_<country>.csv`. **State**
`./formula_of_populism.db`, table `populism`.

- Creates `formula_of_populism_analysis`, `formula_of_populism_us`,
  `formula_of_populism_frontier`.
- Cache key is the **single** column `video_file`, matched against the row's
  `video_filename`. This is why stage 4 must run first: `video_filename` is a
  postprocess-derived key.
- The model is called with a JSON schema (`FormulaOfPopulism`) and a prompt built
  on Laclau's chains of equivalence/antagonism and Palonen's formula of populism
  — the only stage permitted to do political/discourse analysis.
- Structured output format: each element is written as
  `<populism_element>^<populism_affect>`, one per line, per list.

---

## 4. Final dataframe contract

The final `ep24_<country>.csv` is the output other research stages consume.

### Required input columns (must exist in `csv/tiktok_videos.csv`)

`language`, `authorUniqueId`, `scrapedCountry`, `videoId`, `authorNickname`,
`authorSignature`, `videoCreated`, `videoDuration`, `videoDiggCount`,
`videoShareCount`, `videoCommentCount`, `videoPlayCount`, `videoDescription`.

`whisperResult` is **optional on input** and is preserved when present.

### Fields added by each stage

| Stage | Adds |
|---|---|
| preprocess | `whisperResult`, `frame_files`, `ocr_1`, `ocr_2`, `ocr_3`, `ocr_4`, `ocr_5`, `ocr_6`, `whisper_transcript`, `whisper_language`, `whisper_translated` |
| frame | `frame_analysis_1`, `frame_analysis_2`, `frame_analysis_3`, `frame_analysis_4`, `frame_analysis_5`, `frame_analysis_6` |
| summary | `metadata`, `summary_analysis` |
| postprocess | `video_filename`, `entities`, `topics`, `positive`, `neutral`, `negative` |
| populism | `formula_of_populism_analysis`, `formula_of_populism_us`, `formula_of_populism_frontier` |

The issue requires the final Roihu dataframe to retain **all** of these plus new
Phase 2 fields. Legacy fields are never renamed or dropped.

### Field format conventions

These are machine-parsed downstream. Preserve them exactly.

| Field | Format |
|---|---|
| `frame_files` | comma-separated paths |
| `frame_analysis_<n>` | `### **Frame <n> at <seconds> seconds**:` then the model text, embedded in the CSV cell |
| `ocr_<n>` | newline-joined OCR strings |
| `entities`, `topics`, `positive`, `neutral`, `negative` | `, `-joined unique strings |
| `formula_of_populism_us` / `_frontier` | newline-joined `<element>^<affect>` pairs |
| `metadata` | fixed multi-line labelled block (Author name, Author username, …) |

---

## 5. Join keys

| Relation | Key |
|---|---|
| Stages 1–3 SQLite caches | `(author_username, video_id)` |
| Stage 4 → stage 5 | `video_filename` = `authorUniqueId/videoId` (postprocess-derived) |
| Stage 5 cache | `video_file` |

There is **no single stable ID carried across all five stages**. `videoId` alone,
`authorUniqueId` alone and the derived `video_filename` are each used at
different points. Any new stage (DNA, SNA, RDF) that must join back to these rows
has to adopt this existing key set rather than inventing a new identifier, or it
will not join reproducibly.

---

## 6. Failure and skip semantics to preserve

- **Per-row failures do not stop a run.** Every stage wraps per-row work in
  `try/except`, logs, and continues.
- **Row drops are stage-specific and must not be moved.** Stage 1 drops nothing;
  stages 2 and 3 drop on `whisperResult` (and stage 2 additionally on
  `frame_files`); stage 4 skips blank-`summary_analysis` rows; stage 5 skips
  nothing but reads a derived key.
- **A missing video file is not fatal** — the row is written with empty
  multimodal fields and logged as an error.
- **Caches short-circuit work.** A database hit means the model is not called and
  media is not read. Cache-hit and cache-miss paths must produce the same
  columns.

---

## 7. Known inconsistencies to preserve deliberately

These look like defects. They are recorded so the migration does not "tidy" them
incidentally; each is either intentional or requires an explicit human decision.

1. **`formula_of_populism.db` and `formula.log` sit in the CWD**, while the other
   per-stage databases live under `./database/` and the other logs under
   `./logs/`.
2. **`puhti_populism.py` uses the bare names `formula.log` and
   `formula_of_populism.db`** rather than a path prefix.
3. **`en`/`bg` stage-list asymmetry** (§2).
4. **Finnish is absent from the EasyOCR reader list** while present in the stage
   language list (§2).
5. **`puhti_summary.py` hard-codes the frame timestamps** (`0/30/60/90/120/150`)
   rather than deriving them from the extracted frames.
6. **`puhti_frame.py` re-runs its `SELECT` after `fetchone()`** rather than
   reusing the fetched tuple.
7. **`puhti_populism.py` writes `ep24_<country>.csv` on the cache-hit path too**,
   so the output file is rewritten on every run.

---

## 8. What must not change during the Roihu migration

Per `AGENTS.md` and the issue's safety rails, the following are **not**
infrastructure and require an explicit human task to change:

- the five-stage order and the artifacts handed between stages;
- any prompt text;
- any structured-output schema;
- any CSV column name or the format conventions in §4;
- the SQLite table/column layout and cache keys;
- the language/country lists and their asymmetries;
- the drop/skip/failure semantics in §6.

Model selection is the one thing the Roihu baseline deliberately made
configurable (`LACLAUGPT_MULTIMODAL_MODEL`). That is a configuration change, not
an analytical one.

---

## 9. Open questions for the author

1. **New stages and the join key.** Should DNA/SNA/RDF rows key on
   `video_filename` (the stage-5 convention) or on `(authorUniqueId, videoId)`?
   §5 shows there is no existing all-stage key, so this needs a decision rather
   than an assumption.
2. **`en`/`bg`.** Is `bg` in stage 5 but absent from stages 1–4 intentional (a
   country handled only downstream), or an omission?
3. **Language-list source.** Should the per-stage lists eventually come from
   configuration, or stay literal to preserve byte-identical behaviour?
4. **`ep24_<language>.csv` ownership.** It is currently written by stage 4 and
   consumed by stage 5. A new postprocessing stage must not repoint stage 5's
   input without a compatibility note.
