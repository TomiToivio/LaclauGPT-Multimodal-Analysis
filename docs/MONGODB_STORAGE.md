# EP2024 distributed research storage on CSC Roihu

This document defines the storage architecture for the active EP2024 reprocessing work on CSC Roihu.

## Canonical roles

- **Pandas CSV/DataFrames**: canonical human-readable input/output and legacy-compatible research interchange.
- **MongoDB**: shared durable research record for analysis rows, codebooks, memory, RAG, researcher notes, graph/RDF material, embeddings, provenance and backups.
- **Redis**: transient distributed coordination, cache, messaging, locks, job/status information and similar ephemeral state.
- **CSC Allas**: video/media object storage. Videos are downloaded on demand using their stored URL/object identifier after the user configures Allas with `allas_conf`.
- **SQLite / DuckDB**: optional local helpers, compatibility artifacts, imports/exports, temporary indexes and job-local checkpoints. They are not the canonical shared distributed backend.
- **PostgreSQL**: not used for this EP2024 reprocess architecture.

Do not run MongoDB or Redis servers on Roihu compute nodes. Roihu jobs connect to the configured shared services.

## Private configuration

Public code and documentation contain variable names only. The real values are stored in the private companion repository:

`TomiToivio/LaclauGPT-Private/analysis/ep24_reprocess/.env`

Expected variables:

```bash
LACLAUGPT_MONGODB_URI=...
LACLAUGPT_MONGODB_DATABASE=...
LACLAUGPT_REDIS_URL=...
```

Never copy those live values into this public repository, logs, tests, issue comments or documentation.

## Collection naming

Every country-specific collection uses:

```text
laclaugpt_ep2024_reprocess_<country_name>_<collection_name>
```

Examples:

```text
laclaugpt_ep2024_reprocess_finland_dataframe
laclaugpt_ep2024_reprocess_finland_codebooks
laclaugpt_ep2024_reprocess_finland_memory
laclaugpt_ep2024_reprocess_finland_rag
laclaugpt_ep2024_reprocess_finland_research_notes
laclaugpt_ep2024_reprocess_finland_rdf
laclaugpt_ep2024_reprocess_finland_dna
laclaugpt_ep2024_reprocess_finland_sna
laclaugpt_ep2024_reprocess_finland_provenance
laclaugpt_ep2024_reprocess_finland_backup
```

Use the same scheme for all actual EP2024 countries. Shared EU-wide/common material may use an explicitly documented common namespace.

Collection purpose names should remain stable once data has been written.

## MongoDB as document, graph and vector-capable research storage

MongoDB is the canonical durable store even when different access patterns are needed.

### Documents

Store complete research records with stable IDs, source URLs, country/language, stage/run/model provenance and version information. Human-authored material must remain distinguishable from model-generated proposals.

### Graph / RDF

Represent graph material honestly as documents, for example:

- node documents with stable IDs and type/provenance;
- edge documents with source, target, relation and evidence;
- RDF-style subject/predicate/object documents;
- DNA/SNA statements and derived edges.

MongoDB is not described as a native property-graph database merely because graph-shaped documents are stored in it.

### Vectors / RAG

Store:

- source/chunk ID;
- original text;
- English translation where applicable;
- embedding vector;
- embedding model/version;
- country/language;
- provenance and timestamps.

If the deployed MongoDB supports suitable vector search, use it behind an adapter. Otherwise retrieve embeddings/documents from MongoDB and perform similarity search in application code. A local Chroma or other helper may be evaluated only as an optional accelerator/index; it must not become a second canonical research record.

## Redis

Redis is intentionally non-authoritative. Good uses include:

- job coordination and locks;
- cache;
- messaging;
- worker/run status;
- short-lived retrieval caches;
- deduplication or rate-limit state;
- optional queues.

Durable memory, researcher notes, accepted codebooks and final analysis results belong in MongoDB and/or versioned CSV artifacts, not only Redis.

## CSV/DataFrame contract

MongoDB does **not** replace the dataframe.

Every EP2024 run must continue to read/write Pandas-compatible CSV and preserve the legacy dataframe schema. New fields are additive.

Requirements:

1. preserve every legacy field and its historical meaning;
2. keep original-language evidence and English translations side by side;
3. add structured JSON fields when useful;
4. also add a **human-readable Markdown-formatted summary for every major analytical step**;
5. make the final CSV understandable without requiring MongoDB access;
6. support deterministic MongoDB -> DataFrame -> CSV export and CSV -> DataFrame -> MongoDB import;
7. preserve stable row/source/video identifiers across round trips.

Recommended new fields use stable stage-specific names such as `<stage>_summary_md`, plus structured companions such as `<stage>_json` where needed.

## CSC Allas video flow

The dataframe's Allas/public object URL is the source-media key.

Roihu workflow:

1. user configures Allas with `allas_conf`;
2. the job reads the video URL/object identifier from the dataframe;
3. only required videos are downloaded to job-local/project scratch storage;
4. analysis runs locally;
5. derived results go to CSV + MongoDB;
6. provenance retains the original URL/object identifier;
7. video binaries are not stored in MongoDB or Redis.

## Backups and reproducibility

MongoDB is the shared durable research backend, but reproducibility must not depend on one live database instance.

Maintain:

- versioned CSV exports;
- collection/run manifests;
- model/prompt/codebook versions;
- stable source/video IDs;
- timestamps and run IDs;
- backup/export procedures;
- optional Allas/project-storage copies of exported research artifacts.

Never silently overwrite researcher-reviewed material. Human locks/overrides must be explicit and auditable.

## Optional local technologies

SQLite, DuckDB, Parquet and local vector indexes are allowed where they simplify Roihu batch processing. They are implementation aids, not replacements for the canonical MongoDB/CSV architecture.

Evaluate Chroma or a dedicated graph engine only if there is a demonstrated benefit, it works reliably on Roihu, and it does not make the pipeline harder to inspect or reproduce.
