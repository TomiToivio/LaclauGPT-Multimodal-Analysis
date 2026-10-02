# EP24 RDF export (`roihu_rdf.py`)

Stage 12 of the target pipeline: a new, **appended** export stage that projects the
five legacy stages' CSV output into RDF/Turtle. It runs after them, changes no
legacy behaviour, and is read-only with respect to every existing artifact.

```text
preprocess → frame → summary → postprocess → populism → [rdf]
```

## Design rules

1. **RDF is a projection, never the canonical source of truth.** The CSV remains
   authoritative; the graph is derived from it and can be regenerated at any time.
2. **Every predicate maps back to a legacy column.** There is no independent
   ontology — the mapping is the table below, and the code follows it.
3. **Nothing is dropped.** Any column the export does not recognise is still
   emitted, under its exact legacy name, as an `lg:LegacyField` node.
4. **No manufactured relations.** Every relation records whether it is
   `observed` (a fact the platform or a legacy stage recorded) or `model_derived`
   (a model's analysis). Nothing is inferred beyond that.
5. **Stdlib only.** Turtle is written directly, so the stage cannot break a
   multi-hour batch job by failing to import a library.
   `rdflib` is used only by the tests, to validate that the output is genuine RDF.

## Namespaces

| Prefix | IRI | Used for |
|---|---|---|
| `lg` | `https://w3id.org/laclau` + `gpt/` | the project vocabulary, **reused verbatim** from the sibling `LaclauGPT-Data-Analysis` implementation so both graphs share one vocabulary instead of inventing a second |
| `rdf` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` | `rdf:type` — class membership |
| `prov` | `http://www.w3.org/ns/prov#` | `prov:wasGeneratedBy` |
| `dcterms` | `http://purl.org/dc/terms/` | `dcterms:created`, `dcterms:source` |
| `schema` | `https://schema.org/` | `schema:url`, `schema:datePublished`, `schema:identifier` |
| `skos` | `http://www.w3.org/2004/02/skos/core#` | `skos:prefLabel` on vocabulary terms |
| `xsd` | `http://www.w3.org/2001/XMLSchema#` | literal datatypes |

Classes are asserted with **`rdf:type`**, not a private predicate, so SPARQL
queries and reasoners can see them.

## Classes

| Class | Represents | Source |
|---|---|---|
| `lg:Document` | one video/post | one CSV row |
| `lg:Actor` | the posting account | `authorUniqueId` (+ `authorNickname`, `authorSignature`) |
| `lg:Frame` | an extracted keyframe file | active `frame_file`; historical `frame_files` fallback |
| `lg:FrameAnalysis` | a model's reading of a frame | active `frame_analysis_1`; historical `frame_analysis_1..6` fallback |
| `lg:ScreenText` | on-screen text observed in a frame (OCR) | active `ocr_1`; historical `ocr_1..6` fallback |
| `lg:Transcript` | speech transcript | active `asr_transcript`, `asr_translated`, `asr_language`, backend/model provenance; historical Whisper fallback |
| `lg:Summary` | the summary analysis | `summary_analysis` (+ `metadata` as its prompt input) |
| `lg:Topic` | a topic | `topics` |
| `lg:Entity` | an entity | `entities` |
| `lg:SentimentTarget` | a sentiment target | `positive` / `neutral` / `negative` |
| `lg:SentimentAssessment` | a reified link: document → polarity → target | the same three columns |
| `lg:PopulismAnalysis` | the Laclau/Palonen reading | `formula_of_populism_analysis` |
| `lg:PopulismElement` | one `element` with its `affect` | `formula_of_populism_us`, `_frontier` |
| `lg:LegacyField` | any column not otherwise typed | every other column |
| `lg:DNA*` / `lg:SNA*` | network projection, when those stages exist | their node/edge tables |

## Why polarity is reified

Sentiment is emitted as an `lg:SentimentAssessment` node between the document and
the target rather than as a property of the target. The same phrase can be
positive in one document and negative in another, so polarity belongs to the
document's treatment of it, not to the phrase. Collapsing it onto the target would
make two contradictory observations overwrite each other.

## Provenance

Every generated node carries:

| Predicate | Meaning |
|---|---|
| `lg:derivation` | `observed` or `model_derived` |
| `lg:stage` | the pipeline stage that produced it (`preprocess`, `frame`, …, `rdf`) |
| `lg:model` | the model, on model-derived nodes only |
| `lg:runId` | the Slurm job id, or a local run id off-cluster |
| `prov:wasGeneratedBy` | an `lg:activity/<stage>/<runId>` node |
| `dcterms:created` | UTC timestamp of the export |

This is what lets a reader tell a platform-recorded fact from a model's claim,
which the SNA and RDF requirements need in order to distinguish directly observed
social edges from discourse-derived and model-inferred ones.

## Legacy column coverage

Every column the five stages produce. Recognised columns get a typed predicate;
unrecognised ones become `lg:LegacyField` and are **not** dropped.

| Stage | Columns |
|---|---|
| preprocess | active: `frame_file`, `frame_timestamp_seconds`, `ocr_1`, `ocr_backend`, `ocr_model`, `asr_transcript`, `asr_language`, `asr_translated`, `asr_backend`, `asr_model`, runtime/status/provenance fields; frozen legacy rows may still contain `frame_files`, `ocr_2..6`, `whisperResult`, `whisper_*` |
| frame | `frame_analysis_1..6` |
| summary | `metadata`, `summary_analysis`, `authorNickname`, `authorSignature`, `videoCreated`, `videoDescription`, `videoDuration`, `videoCommentCount`, `videoDiggCount`, `videoPlayCount`, `videoShareCount` |
| postprocess | `entities`, `topics`, `positive`, `neutral`, `negative` |
| populism | `formula_of_populism_analysis`, `formula_of_populism_us`, `formula_of_populism_frontier` |
| identity | `authorUniqueId`, `videoId`, `video_filename`, `language`, `scrapedCountry` |

Issue #128 makes the active preprocess representation singular: one `frame_file`
at original t=1.0s, one `ocr_1`, and backend-neutral `asr_*` fields. The RDF
export still reads the two historical `frame_files` encodings and Whisper-named
fields so frozen pre-#128 artifacts remain exportable; it never writes those
legacy fields back into the dataframe.

## Usage

Runs from the private runtime root, like the other stages:

```bash
python roihu_rdf.py                        # every configured language
python roihu_rdf.py --language fi          # one language
python roihu_rdf.py --sample 5 --debug     # smoke test, verbose
python roihu_rdf.py --dry-run              # parse and report, write nothing
python roihu_rdf.py --output-dir ./rdf     # where the output goes (default ./rdf)
```

No model and no GPU are needed, so this stage does **not** require the Ollama
setup in `multimodal_roihu.sbatch`. It reads CSV and writes text.

## Outputs

| File | Contents |
|---|---|
| `rdf/ep24_<language>.ttl` | the RDF/Turtle projection |
| `rdf/summary_<language>.txt` | human-readable per-document summary |

The summary exists so a researcher can read what the graph says about each item
without opening an RDF tool — the same requirement the issue places on the DNA and
SNA layers.

## Adding it to a batch run

The stage is deliberately **not** wired into `scripts/roihu/run_pipeline.sh` yet:
that wrapper's stage list is covered by a baseline test asserting the historical
five-stage order, and the issue asks for small, separately reviewed changes. Add it
explicitly when wanted:

```bash
export LACLAUGPT_MULTIMODAL_STAGES='postprocess populism'
bash scripts/roihu/run_pipeline.sh
python roihu_rdf.py --language fi
```

No model, no GPU, so it can also run on a login node for a quick look.

## Relationship to DNA and SNA

The issue orders DNA and SNA **before** RDF, and those stages are separate,
still-unclaimed work. `maybe_emit_network()` therefore looks for a minimal
conventional shape — `<kind>/<kind>_nodes_<lang>.csv` with an `id` column, and
`<kind>/<kind>_edges_<lang>.csv` with `source`/`target` — and **skips cleanly with a
log line when they are absent**. It is marked provisional in the code: when the
real stages land, this adapter should be replaced with one that reads their actual
contract rather than a guessed one. Nothing is emitted today, so the export is
complete and correct for the five legacy stages as they stand.

## Privacy

Reads only pipeline output, writes only Turtle and text. No network access, no
credentials. The tests use a synthetic fixture with fabricated values and are the
only committed example data.
