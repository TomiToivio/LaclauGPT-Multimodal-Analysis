# EP24 Steps 1–9 cumulative pipeline contract

The active Roihu pipeline follows one rule: **each numbered step enriches the
same logical record instead of replacing it**. CSV/Pandas remains the
human-readable compatibility interface; MongoDB is the durable shared backend
when enabled. SQLite is a job-local/checkpoint compatibility helper.

## Canonical identity

Every stage preserves the source identity fields, especially:

- `video_id` as an opaque string;
- `allas_filename` as the authoritative media key;
- `country`, `source_type`, `source_recording`, `author_username`;
- `_storage_id` when bootstrap/storage orchestration has supplied it.

Downstream graph/export stages use `_storage_id` when available and otherwise
derive deterministic identity from the same canonical source record. They must
not generate a new random content identity.

## Cumulative stage contract

| Step | Adds | Persistence rule |
|---|---|---|
| 1 preprocess | ASR/OCR/keyframe/preprocess fields | preserve every input column |
| 2 frame | deep keyframe analysis | additive |
| 3 video | whole-video analysis + scroll QA | additive |
| 4 summary | multimodal fusion/summary | additive |
| 5 postprocess | normalized entities/themes/sentiment targets | additive |
| 6 discourse | Laclau/Palonen analysis | additive |
| 7 DNA | `dna_*` statements + Markdown | additive |
| 8 SNA | `sna_nodes_json`, `sna_edges_json`, metrics, Castells interpretation + Markdown | additive + graph exports |
| 9 RDF | `rdf_*` export metadata + RDF files | deterministic projection only |

The machine-readable list of expected fields is
`config/ep24_pipeline_columns.json`.

## Step 8 SNA contract

Step 8 deliberately implements basic SNA rather than an advanced network-science
stack.

The empirical graph is explicit Node–Edge–Node data:

- node: `node_id`, `node_type`, `label`, country/platform/source identity,
  timestamp and provenance item IDs;
- edge: `edge_id`, `source`, `target`, `edge_type`/`relation_type`,
  direction, evidence quote, country/platform/source identity and provenance.

Stable upstream entity IDs are preferred. When entity resolution has no ID,
Step 8 hashes the normalized label deterministically.

Basic metrics only: degree, in/out degree, weighted degree, weak components,
density and simple degree centrality. These are calculated deterministically
after relation extraction.

The LLM relation extractor may only admit evidence-supported relations. The
Castells prompt runs **after topology is fixed** and cannot add or remove
nodes/edges. Its output is stored separately as
`sna_castells_interpretation_markdown` and embedded in the human-readable
`sna_analysis_markdown` report.

Step 8 also writes sibling machine-readable tables next to the cumulative CSV:

- `<output>.sna_nodes.csv`
- `<output>.sna_edges.csv`

When MongoDB is enabled, the cumulative dataframe document is patched rather
than replaced and graph records are mirrored to the country-specific `sna`
collection using stable graph IDs.

## Step 9 RDF contract

RDF is a final projection, never a competing source of truth. Step 9 retains
lossless CSV cells and additionally projects queryable resources for:

- content/document and CSV-row provenance;
- accounts/authors;
- normalized/source entities and themes;
- Laclau coding already present in the cumulative record;
- SNA nodes, edges, metrics and their source-row provenance.

SNA RDF resources use the Step 8 `node_id` and `edge_id` as their stable
identity keys. Step 9 therefore serializes the graph created by Step 8 rather
than reconstructing a different network.

The exporter writes deterministic N-Triples plus an equivalent `.ttl` copy.
Tests independently parse the RDF with RDFLib when available.

## CLI and orchestration

All numbered entry points keep the common `--country/-c` and `--limit/-n`
contract through `ep24_cli.configure_step_cli`. A plain command such as

```bash
python step_8_roihu_social_network_analysis.py
```

continues to work. Slurm wrappers may set
`LACLAUGPT_INPUT_CSV`/`LACLAUGPT_OUTPUT_CSV` for the cumulative checkpoint
chain.

## Public/private boundary

Real data, codebooks, credentials, private settings and researcher material stay
in `LaclauGPT-Private` or private CSC storage. Public graph/RDF code contains
only schemas, interfaces and synthetic fixtures.
