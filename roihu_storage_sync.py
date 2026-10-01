"""Optional post-pipeline sync of EP24 CSV outputs into external MongoDB."""

from __future__ import annotations

import glob
import os
import sys
import uuid
from pathlib import Path

from roihu_storage import MongoStorage, StorageConfig, jsonl_documents, sqlite_memory_documents


def main() -> int:
    config = StorageConfig.from_env()
    if not config.mongo_enabled:
        print("MongoDB storage disabled; CSV-only mode remains active.")
        return 0

    pattern = os.getenv("LACLAUGPT_STORAGE_INPUT_GLOB", "csv/*.csv")
    run_id = os.getenv("LACLAUGPT_RUN_ID") or os.getenv("SLURM_JOB_ID") or str(uuid.uuid4())
    model = os.getenv("LACLAUGPT_MULTIMODAL_MODEL")

    storage = MongoStorage(config)
    try:
        paths = sorted(Path(p) for p in glob.glob(pattern))
        if not paths:
            print(f"No CSV files matched {pattern!r}; nothing to persist.")
            return 0
        total = 0
        for path in paths:
            count = storage.csv_import(
                path,
                purpose="analysis",
                stage="pipeline_complete",
                run_id=run_id,
                model=model,
            )
            total += count
            print(f"Persisted {count} rows from {path} -> {config.collection('analysis')}")
        memory_db = os.getenv("LACLAUGPT_MEMORY_DB")
        if memory_db:
            memory_docs = sqlite_memory_documents(memory_db, config=config)
            memory_count = storage.upsert_documents("memory", memory_docs)
            total += memory_count
            print(f"Persisted {memory_count} reviewed memory records -> {config.collection('memory')}")

        rag_jsonl = os.getenv("LACLAUGPT_RAG_JSONL")
        if rag_jsonl:
            rag_docs = jsonl_documents(rag_jsonl, config=config, purpose="rag")
            rag_count = storage.upsert_documents("rag", rag_docs)
            total += rag_count
            print(f"Persisted {rag_count} RAG records -> {config.collection('rag')}")

        print(f"MongoDB sync complete: {total} records, prefix={config.prefix}")
        return 0
    finally:
        storage.close()


if __name__ == "__main__":
    sys.exit(main())
