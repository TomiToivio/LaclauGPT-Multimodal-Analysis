# External MongoDB storage on CSC Roihu

Issue #12 adds an optional persistent storage layer around the EP24-compatible pipeline. The legacy CSV workflow remains supported and MongoDB does not replace raw media storage.

## Storage roles

- CSC project storage / Allas: raw media and large files.
- Pandas CSV/DataFrames: researcher-readable import, exchange and stage outputs.
- External MongoDB: durable structured analysis state, memory, RAG, entities, provenance and structured backups.
- Redis: may be used for cache/coordination elsewhere, but is not durable memory here.

MongoDB is external. Do not run a MongoDB server on a Roihu compute node.

## Configuration

Public code contains no credentials. Supply secrets from LaclauGPT-Private, private CSC storage, or the job environment.

    export LACLAUGPT_MONGO_ENABLED=1
    export LACLAUGPT_MONGO_URI='mongodb+srv://...'
    export LACLAUGPT_MONGO_DATABASE=laclaugpt
    export LACLAUGPT_DATASET=ep24
    export LACLAUGPT_COUNTRY=fi

Never commit a real URI, certificate, password or token.

With LACLAUGPT_MONGO_ENABLED=0, which is the default, the new storage stage is a no-op and the pipeline remains CSV-only.

## Collection naming

Collections are generated, never hard-coded:

    laclaugpt_<dataset>_<country>_<purpose>

For Finland:

    laclaugpt_ep24_fi_memory
    laclaugpt_ep24_fi_rag
    laclaugpt_ep24_fi_analysis
    laclaugpt_ep24_fi_entities
    laclaugpt_ep24_fi_backup

For Poland the same configuration automatically produces laclaugpt_ep24_pl_memory, laclaugpt_ep24_pl_rag, and so on. A new EP24 country only requires changing LACLAUGPT_COUNTRY.

## Roihu pipeline

The default compatibility sequence is:

    preprocess -> frame -> summary -> postprocess -> enrich -> populism -> storage

The final stage calls roihu_storage_sync.py. When MongoDB is disabled it exits successfully without changing CSV files. When enabled, it imports CSVs matching LACLAUGPT_STORAGE_INPUT_GLOB, default csv/*.csv, into the country-specific analysis collection using upsert semantics.

Submit normally after exporting private configuration:

    sbatch --account="$CSC_ACCOUNT" scripts/roihu/multimodal_roihu.sbatch

The job connects outward to the configured MongoDB. Connection failure is explicit when MongoDB mode is requested.

## DataFrame and CSV API

roihu_storage.py supports DataFrame to MongoDB, MongoDB to DataFrame, CSV to DataFrame to MongoDB, and MongoDB to DataFrame to CSV.

Example:

    from roihu_storage import MongoStorage, StorageConfig

    store = MongoStorage(StorageConfig.from_env())
    store.dataframe_upsert("analysis", df, stage="summary", run_id="123")
    df2 = store.dataframe_export("analysis")
    store.csv_import("csv/ep24_fi.csv", purpose="analysis", run_id="123")
    store.csv_export("exports/ep24_fi_from_mongo.csv", purpose="analysis")
    store.close()

Rows carry a stable _storage_id. Existing identifiers such as id, post_id, video_id, document_id, source_id or url are preferred; otherwise a deterministic source/row/content fingerprint is used. Mongo exports preserve _storage_id, so repeated round-trips remain idempotent.

Legacy columns are not renamed or removed. Mongo-specific provenance is additive.

## Memory

Storage(...).memory writes to <prefix>_memory. Records can contain source IDs, memory type, text, structured metadata, pipeline stage/model provenance, references, embeddings and version/supersession metadata. The facade adds dataset, country and timestamp and uses upsert-based stable IDs.

MongoDB memory is complementary to the existing reviewed SQLite memory workflow. SQLite remains useful as a controlled snapshot/review artifact; external MongoDB provides cross-job durable state where enabled.

## RAG

Storage(...).rag writes to <prefix>_rag. Store canonical source/chunk IDs, original and translated English text, language, source metadata, embeddings, embedding model/version and run provenance.

The interface hides MongoDB details:

    storage.rag.upsert(record)
    hits = storage.rag.retrieve("European Parliament Finland")

retrieve currently provides a portable lexical fallback. Deployments with MongoDB vector-search capability can replace that implementation behind the same interface while retaining stored embeddings and audit metadata.

## Backup and recovery

MongoDB is an additional structured-data persistence/backup target, not a raw media backup. For research-friendly recovery, export collections back to CSV using csv_export and copy those files to durable CSC storage/Allas according to the project's data-management plan.

Never overwrite research exports silently. Write timestamped/versioned destinations or explicitly review replacements.

For full database-native backup/restore, use the external MongoDB deployment's approved mongodump/mongorestore or managed backup mechanism from an appropriate host, not by starting database services on Roihu compute nodes.

## Dependencies and testing

Install Mongo support explicitly:

    python -m pip install -e '.[mongo]'

CSV-only mode does not import pymongo.

Tests use an in-memory fake backend and never connect to the real external MongoDB. They cover collection prefixing, FI/PL isolation, DataFrame mapping/export, idempotent upserts, disabled mode, failure behavior, memory persistence and RAG retrieval.
