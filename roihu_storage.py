"""External MongoDB storage plus Pandas/CSV interoperability for Roihu.

MongoDB is optional. The module imports pymongo only when Mongo-backed storage is
actually requested so CSV-only workflows remain usable without that dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

_PREFIX_RE = re.compile(r"^[a-z0-9_]+$")
_DEFAULT_PURPOSES = ("memory", "rag", "analysis", "entities", "backup")


def _slug(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("storage name component must not be empty")
    return value


def collection_prefix(dataset: str, country: str) -> str:
    prefix = f"laclaugpt_{_slug(dataset)}_{_slug(country)}"
    if not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(f"invalid MongoDB collection prefix: {prefix}")
    return prefix


@dataclass(frozen=True)
class StorageConfig:
    dataset: str
    country: str
    mongo_enabled: bool = False
    mongo_uri: str | None = None
    mongo_database: str = "laclaugpt"
    mongo_timeout_ms: int = 5000

    @property
    def prefix(self) -> str:
        return collection_prefix(self.dataset, self.country)

    def collection(self, purpose: str) -> str:
        return f"{self.prefix}_{_slug(purpose)}"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        enabled = os.getenv("LACLAUGPT_MONGO_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        return cls(
            dataset=os.getenv("LACLAUGPT_DATASET", "ep24"),
            country=os.getenv("LACLAUGPT_COUNTRY", "fi"),
            mongo_enabled=enabled,
            mongo_uri=os.getenv("LACLAUGPT_MONGO_URI"),
            mongo_database=os.getenv("LACLAUGPT_MONGO_DATABASE", "laclaugpt"),
            mongo_timeout_ms=int(os.getenv("LACLAUGPT_MONGO_TIMEOUT_MS", "5000")),
        )


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, (dict, list, tuple, str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(data: Mapping[str, Any]) -> str:
    cleaned = {str(k): _clean_value(v) for k, v in data.items()}
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_record_id(
    record: Mapping[str, Any],
    *,
    dataset: str,
    country: str,
    source_hint: str = "",
    row_hint: str = "",
) -> str:
    for key in ("_storage_id", "id", "post_id", "video_id", "document_id", "source_id", "url"):
        value = record.get(key)
        if value not in (None, ""):
            raw = f"{dataset}|{country}|{key}|{value}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raw = f"{dataset}|{country}|{source_hint}|{row_hint}|{_canonical_json(record)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dataframe_to_documents(
    df: pd.DataFrame,
    *,
    config: StorageConfig,
    stage: str,
    run_id: str,
    source_hint: str = "",
    model: str | None = None,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    docs: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        values = {str(k): _clean_value(v) for k, v in row.to_dict().items()}
        storage_id = stable_record_id(
            values,
            dataset=config.dataset,
            country=config.country,
            source_hint=source_hint,
            row_hint=str(index),
        )
        values["_storage_id"] = storage_id
        values["_provenance"] = {
            "dataset": config.dataset,
            "country": config.country,
            "pipeline_stage": stage,
            "run_id": run_id,
            "timestamp": now,
            "source": source_hint or None,
            "model": model,
        }
        docs.append(values)
    return docs


def documents_to_dataframe(documents: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in documents:
        row = dict(doc)
        row.pop("_id", None)
        rows.append(row)
    return pd.DataFrame(rows)


def sqlite_memory_documents(path: str | Path, *, config: StorageConfig) -> list[dict[str, Any]]:
    """Export reviewed canonical EP24 SQLite objects into durable Mongo memory documents."""
    db_path = Path(path)
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='objects'"
        ).fetchone()
        if not exists:
            return []
        rows = list(db.execute("SELECT * FROM objects WHERE state='CANONICAL' ORDER BY obj_id"))
    docs = []
    for row in rows:
        payload = dict(row)
        obj_id = str(payload["obj_id"])
        docs.append(
            {
                "_storage_id": stable_record_id(
                    {"source_id": obj_id},
                    dataset=config.dataset,
                    country=config.country,
                    source_hint="sqlite_memory",
                ),
                "source_id": obj_id,
                "dataset": config.dataset,
                "country": config.country,
                "memory_type": payload.get("kind", ""),
                "text": payload.get("canonical_label", ""),
                "structured_metadata": payload,
                "originating_pipeline_stage": "reviewed_sqlite_memory",
                "provenance": {
                    "backend": "sqlite",
                    "sqlite_path": str(db_path),
                    "origin": payload.get("origin", ""),
                },
                "version": payload.get("updated_at", ""),
            }
        )
    return docs


def jsonl_documents(path: str | Path, *, config: StorageConfig, purpose: str) -> list[dict[str, Any]]:
    """Load newline-delimited records for optional RAG/memory handoff between pipeline stages."""
    source = Path(path)
    if not source.exists():
        return []
    docs = []
    with source.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{row_number}: expected a JSON object")
            record.setdefault("dataset", config.dataset)
            record.setdefault("country", config.country)
            record.setdefault(
                "_storage_id",
                stable_record_id(
                    record,
                    dataset=config.dataset,
                    country=config.country,
                    source_hint=str(source),
                    row_hint=str(row_number),
                ),
            )
            record.setdefault("provenance", {"source": str(source), "purpose": purpose})
            docs.append(record)
    return docs


class MongoStorage:
    """Thin storage boundary used by memory, RAG and analysis facades."""

    def __init__(self, config: StorageConfig, client: Any | None = None):
        if not config.mongo_enabled:
            raise RuntimeError("MongoDB storage requested while LACLAUGPT_MONGO_ENABLED is disabled")
        if not config.mongo_uri and client is None:
            raise RuntimeError("LACLAUGPT_MONGO_URI is required when MongoDB is enabled")
        self.config = config
        self._owns_client = client is None
        if client is None:
            try:
                from pymongo import MongoClient
            except ImportError as exc:
                raise RuntimeError(
                    "MongoDB mode requires pymongo; install the 'mongo' optional dependency"
                ) from exc
            client = MongoClient(
                config.mongo_uri,
                serverSelectionTimeoutMS=config.mongo_timeout_ms,
                connectTimeoutMS=config.mongo_timeout_ms,
                retryWrites=True,
            )
        self.client = client
        try:
            self.client.admin.command("ping")
        except Exception as exc:
            if self._owns_client:
                self.client.close()
            raise RuntimeError(f"external MongoDB connection failed: {exc}") from exc
        self.db = self.client[config.mongo_database]

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def collection_name(self, purpose: str) -> str:
        return self.config.collection(purpose)

    def upsert_documents(
        self, purpose: str, documents: Iterable[Mapping[str, Any]], *, id_field: str = "_storage_id"
    ) -> int:
        docs = [dict(d) for d in documents]
        if not docs:
            return 0
        try:
            from pymongo import ReplaceOne
            operations = [
                ReplaceOne({id_field: d[id_field]}, d, upsert=True)
                for d in docs
                if d.get(id_field) not in (None, "")
            ]
            if operations:
                self.db[self.collection_name(purpose)].bulk_write(operations, ordered=False)
            return len(operations)
        except ImportError:
            # Test/fake clients can implement replace_one without requiring pymongo.
            collection = self.db[self.collection_name(purpose)]
            count = 0
            for doc in docs:
                key = doc.get(id_field)
                if key in (None, ""):
                    continue
                collection.replace_one({id_field: key}, doc, upsert=True)
                count += 1
            return count

    def find(self, purpose: str, query: Mapping[str, Any] | None = None, *, limit: int = 0) -> list[dict]:
        cursor = self.db[self.collection_name(purpose)].find(dict(query or {}))
        if limit:
            cursor = cursor.limit(limit)
        return [dict(item) for item in cursor]

    def dataframe_upsert(
        self,
        purpose: str,
        df: pd.DataFrame,
        *,
        stage: str,
        run_id: str,
        source_hint: str = "",
        model: str | None = None,
    ) -> int:
        documents = dataframe_to_documents(
            df,
            config=self.config,
            stage=stage,
            run_id=run_id,
            source_hint=source_hint,
            model=model,
        )
        return self.upsert_documents(purpose, documents)

    def dataframe_export(self, purpose: str, query: Mapping[str, Any] | None = None) -> pd.DataFrame:
        return documents_to_dataframe(self.find(purpose, query))

    def csv_import(
        self,
        csv_path: str | Path,
        *,
        purpose: str = "analysis",
        stage: str = "csv_import",
        run_id: str = "manual",
        model: str | None = None,
    ) -> int:
        path = Path(csv_path)
        df = pd.read_csv(path)
        return self.dataframe_upsert(
            purpose,
            df,
            stage=stage,
            run_id=run_id,
            source_hint=str(path),
            model=model,
        )

    def csv_export(
        self,
        csv_path: str | Path,
        *,
        purpose: str = "analysis",
        query: Mapping[str, Any] | None = None,
    ) -> Path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe_export(purpose, query).to_csv(path, index=False)
        return path


class _PurposeStore:
    def __init__(self, storage: MongoStorage, purpose: str):
        self.storage = storage
        self.purpose = purpose

    def put(self, record: Mapping[str, Any]) -> int:
        doc = dict(record)
        doc.setdefault("dataset", self.storage.config.dataset)
        doc.setdefault("country", self.storage.config.country)
        doc.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if not doc.get("_storage_id"):
            doc["_storage_id"] = stable_record_id(
                doc,
                dataset=self.storage.config.dataset,
                country=self.storage.config.country,
                source_hint=self.purpose,
            )
        return self.storage.upsert_documents(self.purpose, [doc])

    def search(self, query: Mapping[str, Any] | None = None, *, limit: int = 20) -> list[dict]:
        return self.storage.find(self.purpose, query, limit=limit)


class RAGStore(_PurposeStore):
    def upsert(self, record: Mapping[str, Any]) -> int:
        return self.put(record)

    def retrieve(self, text: str, *, limit: int = 10) -> list[dict]:
        # Portable fallback when Atlas/vector search is unavailable.
        # A deployment-specific vector index can be added behind this interface.
        candidates = self.search(limit=500)
        terms = {t.lower() for t in re.findall(r"\w+", text) if len(t) > 2}
        scored: list[tuple[int, dict]] = []
        for item in candidates:
            haystack = " ".join(
                str(item.get(k, "")) for k in ("text", "original_text", "translated_english")
            ).lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


class Storage:
    def __init__(self, mongo: MongoStorage):
        self.mongo = mongo
        self.memory = _PurposeStore(mongo, "memory")
        self.rag = RAGStore(mongo, "rag")
        self.analysis = _PurposeStore(mongo, "analysis")
        self.entities = _PurposeStore(mongo, "entities")
        self.backup = _PurposeStore(mongo, "backup")

    @classmethod
    def from_env(cls, client: Any | None = None) -> "Storage":
        return cls(MongoStorage(StorageConfig.from_env(), client=client))

    def export_dataframe(self, purpose: str, query: Mapping[str, Any] | None = None) -> pd.DataFrame:
        return self.mongo.dataframe_export(purpose, query)


def supported_collection_names(config: StorageConfig) -> dict[str, str]:
    return {purpose: config.collection(purpose) for purpose in _DEFAULT_PURPOSES}
