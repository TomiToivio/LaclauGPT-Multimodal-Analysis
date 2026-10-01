# EP24 codebook demo: Finland + Poland first

Tracking: issue #33.

This is the first implementation slice of the EP24 bilingual country-codebook work.
The demo deliberately targets **Finland (fi)** and **Poland (pl)** before the
remaining countries.

## Why only FI + PL first

Finland and Poland already have the strongest private researcher-grounded codebook
history and are the intended first Roihu demo datasets. The rest of the country
set remains valid future scope, including Bulgaria, but should not block the demo.

## Active language support

The active Roihu pipeline now includes Bulgarian `bg` in addition to:

`fi, sv, pl, pt, de, es, hu, hr, fr, bg, en`

Country identity is not inferred from language. The country-aware runtime keeps
an explicit mapping, including `BG -> bg`, `FI -> fi`, and `PL -> pl`.

Archived Puhti/legacy scripts are left unchanged unless a separate compatibility
task explicitly requires them.

## FI/PL private inputs

The first demo uses private resources from `LaclauGPT-Private`.

Primary reconstructed sources:

- `analysis/ep24_reprocess/codebook_sources/entities.xlsx`
- `analysis/ep24_reprocess/codebook_sources/persons.xlsx`
- `analysis/ep24_reprocess/codebook_sources/themes.xlsx`
- `analysis/ep24_reprocess/codebook_sources/research_notes.xlsx`

Existing researcher-grounded runtime books under `analysis/ep24/codebooks/`
remain authoritative inputs and must not be overwritten by model discoveries.

The private demo builder writes draft/reviewable FI/PL artifacts from the newer
reconstruction sources without publishing the contents.

## Build the private FI/PL demo codebooks

From the private repository:

```bash
python analysis/ep24_reprocess/scripts/build_demo_fi_pl_codebooks.py
```

The builder reads all four controlled workbook families, including
`persons.xlsx`, and emits only Finland and Poland for this first pass.

## Run only FI + PL on Roihu

For demo jobs, constrain stages that honor `LACLAUGPT_LANGUAGES`:

```bash
export LACLAUGPT_LANGUAGES=fi,pl
```

For older stages with module-level lists, invoke the FI and PL stage calls
explicitly or use the corresponding FI/PL input files. Do not process the full
country set merely because `bg` and the other languages are supported.

## Evidence rule

Codebooks, memory and RAG provide background context and normalization candidates.
They do not prove that a current video contains an actor, theme, grievance,
populist frontier, social-contract claim, or political position.

Synthetic-profile political preference is also context only. It must stay
separate from the evidence-based interpretation of a video/feed item.

## Next demo work

1. validate FI/PL aliases from entities/persons/themes;
2. audit FI/PL research-note fields and provenance;
3. enrich FI/PL with bilingual 2024 election sources;
4. seed FI/PL memory and RAG;
5. add conservative exact/alias/fuzzy candidate resolution;
6. expose codebook context to summary/discourse/postprocess stages without
   destructively changing transcript/OCR/source fields;
7. run the FI/PL demo and compare against legacy outputs.

Other countries follow after the demo path is working.
