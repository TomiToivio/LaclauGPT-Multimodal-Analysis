# EP24 canonical researcher-feed schema

The active EP24 reprocessing pipeline consumes **researcher-recorded social-media
feed clips** that were split from continuous digital-ethnography recordings. It
does not treat scraper exports as the source model.

## Canonical input

Every country uses the same 15 source columns, in this order:

```text
country
author_username
account_type
source_type
source_recording
video_id
sequence_number
political_preference
allas_filename
new_entity
new_theme
video_duration
researcher_new_persons
researcher_new_themes
researcher_note
```

The source files live in the private data repository under
`analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv`.

`video_id` is an opaque string and must never be numerically coerced.
`allas_filename` is the authoritative media key. Do not reconstruct an Allas
path from TikTok scraper fields. `country` and `source_recording` come from
the row and must survive every stage unchanged.

## Cumulative dataframe rule

The CSV/DataFrame is the human-readable research record. Each stage receives
all source columns plus every upstream result and appends its own outputs.
Stages must not rebuild a reduced row or silently discard columns.

Typical growth:

```text
15 source fields
+ Whisper/OCR
+ frame/native-video description
+ multimodal summary
+ postprocess/codebook/memory enrichment
+ Laclau analysis
+ DNA
+ SNA
+ RDF/provenance fields
```

`ep24_pipeline.py` contains the shared load/write/context helpers and
`ep24_schema.py` contains the canonical names and compatibility aliases.

## Provenance in prompts

Prompt context is separated into three classes:

1. **Source metadata**: factual row-level collection context.
2. **Researcher annotation**: human-provided notes/entities/themes/preferences.
3. **Model/enrichment context**: transcripts, OCR, VLM descriptions, summaries,
   codebook/memory/RAG results and downstream analytical fields.

Researcher annotation is not model output. Model-derived analysis is not human
ground truth. Later stages may use both, but the distinction must remain visible.

## Media handling

Every clip originates from a split researcher feed recording. The first 1.0
second is the known scroll transition from the previous item and is excluded
from media analysis. Whole-video analysis also records later splitter failures
using `SCROLL`, `SCROLL_SECONDS` and `needs_resplit`.

For Qwen3-VL through current vLLM, video metadata remains paired with the video
inside `multi_modal_data["video"]`. Moving that metadata out into detached
processor kwargs triggers vLLM's "Video metadata is required but not found in mm
input" error.

## Historical compatibility

Legacy scraper names such as `authorUniqueId`, `videoId`,
`scrapedCountry` and `videoDuration` may be read through aliases for old
files. They are not the canonical schema for new EP24 reprocessing. Historical
Puhti scripts and the `legacy` branch remain preserved for publication
reproducibility.
