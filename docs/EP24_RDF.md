# Lossless EP24 CSV → RDF projection (issue #5)

This additive stage follows the historical sequence: preprocess → frame → summary
→ postprocess → populism. DNA and SNA, when implemented, precede RDF. The frozen
legacy reference is `e011c34274c41923e801c32c824c5afa7468e1a1`.

Design reference: LaclauGPT-Data-Analysis commit
`bf80f435c0390a1d96b3cec21bda09b6c83e7ca0`,
`src/laclaugpt_data_analysis/rdf.py` and `knowledge_graph.py`.
Adapt stable project-scoped identities, PROV activities and explicit assertion
provenance to CSV/SQLite; do not import the AI26 canonical record or service stack.

RDF is a projection, never the authoritative research dataset. Keep every original
CSV field as a literal, including human edits, unknown columns and empty strings.
Interpret only explicitly structured fields. Never split legacy comma-separated
entities into presumed canonical actors or interpret sentiment as DNA agreement.

## Scope and coordination

`roihu_csv_rdf.py` is the lossless CSV/provenance portion of RDF export. The
separately claimed `roihu_rdf.py` semantic graph stage can reuse `export()` or
`project_row()`; this contribution does not claim that entry point, DNA/SNA
extraction, entity normalization or a complete Phase 2 graph ontology. Existing
and future DNA/SNA columns are retained as literal cells, not guessed as edges.
The current legacy scripts and default runner are untouched.

## Run

Python 3.11+ standard library is sufficient; no GPU, network, Ollama, RDF service
or new production dependency is required. Run after populism (and later DNA/SNA).

```bash
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT='/private/runtime'
python /public/checkout/roihu_csv_rdf.py
```

Default input is `ep24_fi.csv` under the private root; default cache is
`database/rdf.sqlite3`; output is a fresh timestamped directory under `rdf/`.
With no environment variable, the working directory is the runtime root.

```bash
python /public/checkout/roihu_csv_rdf.py \
  --input /private/runtime/ep24_pl.csv --dataset-id ep24_pl \
  --output-dir /private/runtime/rdf/polish-pilot --limit 10 --debug
```

Use `--dry-run` for header/configuration checks without writes (it does not validate
every row). `--project` and `--base-uri` scope identities; the default base is
the placeholder `https://data.example/laclaugpt`, not a hosted endpoint.
Set a stable namespace before downstream consumers depend on it.

Batch template (submit from private storage so Slurm logs stay private):

```bash
export LACLAUGPT_MULTIMODAL_PUBLIC_ROOT='/public/checkout'
cd "$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT"
sbatch --account="$CSC_ACCOUNT" --partition="$CSC_CPU_PARTITION" \
  "$LACLAUGPT_MULTIMODAL_PUBLIC_ROOT/scripts/roihu/csv_rdf_roihu.sbatch" \
  --input ep24_fi.csv --dataset-id ep24_fi
```

The allocation and available CPU partition must be supplied for the actual Roihu
environment. The template has not been submitted to Roihu.

## Outputs and researcher edits

| File | Contract |
|---|---|
| `source.csv` | Byte copy of the input used for this run |
| `final.csv` | All original columns, values and row order, plus four RDF columns |
| `graph.nt` | UTF-8 N-Triples; standard RDF parsers can read it |
| `warnings.jsonl` | Row-level identity/structured-field warnings |
| `manifest.json` | Completion marker, source/code hashes, run ID, timestamp, configuration, counts |

Original cell values are retained exactly as strings, including leading zeros,
large IDs, unknown columns, whitespace, multiline notes and empty cells. CSV quoting
and line endings may differ in `final.csv`; the original bytes remain in `source.csv`.

Added columns are `rdf_document_uri`, `rdf_row_uri`, `rdf_schema_version` and
`rdf_status` (`ok` or `warning`). Input using any of these names is rejected rather
than overwritten. Edit the authoritative source CSV and run into a **new** output
directory. Existing output directories are refused, including partial runs, so
manual changes to generated outputs are never silently replaced. All copies,
caches, warnings and manifests contain private research material; keep them private.

## Identity and schema version `ep24-rdf-1`

Documents use project + dataset + first available identity from `document_id`,
`videoId`, `video_filename`, `source_url`, `url`. This is an export-local policy,
not a change to any legacy cache key. All original keys remain available in cells.
Set the same `--dataset-id` across reruns of the same corpus; it defaults to the
input filename stem. Different corpora should have different dataset IDs.

Document URIs survive reorder when the selected source ID is present. Duplicate
source IDs share a document, but each input row has a separate row URI and cells.
Without an ID, the fallback is dataset + row number, with a warning: it is not
stable under reordering. Row/cell URIs identify positions within a dataset, so
graphs from different export snapshots must be kept separate (e.g. named graphs).
Do not union snapshots as if their row cells were timeless assertions.

Namespaced terms use `ep24: = https://w3id.org/laclaugpt/ep24/`. This is a local
EP24 projection profile under the upstream vocabulary, not a claim that its terms
are published or accepted upstream. Standard provenance uses
`prov: = http://www.w3.org/ns/prov#` and `rdf:type`.

| Class / predicate | Meaning |
|---|---|
| `Document`, `CSVRow`, `CSVCell` | Source item, its CSV occurrence, individual cell |
| `document`, `cell`, `column`, `value` | Exact mapping from row to original columns/values |
| `rowNumber`, `identityStrategy` | One-based data row index and chosen ID column |
| `LaclauCoding`, `coding` | Optional structured coding attached to a row |
| `category`, `element`, `affect` | Legacy `us`/`frontier` and `element^affect` pair |
| `assertionKind` | `coded-origin-unspecified`: legacy data may be model or human edited |
| `stageVersion`, `codeSHA256`, `sha256` | Export schema, executing script hash, source file hash |
| `prov:Activity`, `prov:used` | Export run and exact source snapshot |
| `prov:wasDerivedFrom`, `prov:wasGeneratedBy` | Cell→row→file and row→export-run provenance |

Example shape (identifiers shortened here only):

```turtle
@prefix ep24: <https://w3id.org/laclaugpt/ep24/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
<https://example.org/row/1> ep24:cell <https://example.org/cell/1> .
<https://example.org/cell/1> a ep24:CSVCell ;
  ep24:column "researcher_note" ; ep24:value "retain exactly" ;
  prov:wasDerivedFrom <https://example.org/row/1> .
```

Structured Laclau projection only reads the legacy newline-separated
`element^affect` fields. Malformed entries remain intact in raw cells, receive
warnings, and produce no guessed coding. Names, sentiment, frames and free text
remain literal cells. Unknown model/prompt/codebook provenance is not fabricated;
if present as columns, it is preserved. Export-run provenance describes the export,
not the original inference.

## Checkpoints and failure handling

SQLite caches row projections keyed by script hash, schema, namespace, project,
dataset, row position and **every cell value**. Editing a human note or changing
configuration invalidates that row's cache. Cache entries commit after each row.
Rerun into a fresh output directory to reuse completed projections; transcript,
frame and model stages are never invoked. This still scans the CSV and rebuilds
the complete graph/CSV to avoid incomplete appended output.

Malformed analytical pairs are recoverable row warnings. Duplicate/blank CSV
headers, inconsistent row widths, invalid encoding and storage failures stop the
export rather than dropping or misaligning research data. `.partial` outputs and
the cache remain for diagnosis. Only `manifest.json` marks successful completion;
an empty header-only CSV legitimately exports zero rows. `--limit` produces a
sample and is recorded in the manifest. Logs show per-row progress, cache decisions,
warnings, paths and elapsed time; no model call occurs.

## Validation and remaining work

```bash
python -m pip install 'rdflib>=7,<8' 'ruff>=0.11,<1'
python -m unittest discover -s tests -p test_roihu_csv_rdf.py -v
python -m ruff check roihu_csv_rdf.py tests/test_roihu_csv_rdf.py
```

Tests cover exact legacy cell retention, Unicode/control-character RDF roundtrip,
manual edits, cache invalidation, restart after a structural failure, missing and
duplicate IDs, unsafe headers, no-write dry run and overwrite refusal. An independent
CI workflow requires the RDF parser so that check cannot silently skip in CI.

Real EP24/Roihu validation, the semantic DNA/SNA integration, inference provenance,
model audits, codebooks and remaining modernization stages are still outstanding.
This contribution does not complete or close issue #5.
