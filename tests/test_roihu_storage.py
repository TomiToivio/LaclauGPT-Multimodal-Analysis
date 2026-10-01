from __future__ import annotations

from copy import deepcopy
import sqlite3

import pandas as pd
import pytest

from roihu_storage import MongoStorage, Storage, StorageConfig, collection_prefix, dataframe_to_documents, documents_to_dataframe, sqlite_memory_documents


class FakeCursor(list):
    def limit(self, value):
        return FakeCursor(self[:value])


class FakeCollection:
    def __init__(self):
        self.docs = {}

    def replace_one(self, query, doc, upsert=False):
        _, key = next(iter(query.items()))
        assert upsert is True
        self.docs[key] = deepcopy(doc)

    def find(self, query):
        return FakeCursor([
            deepcopy(doc) for doc in self.docs.values()
            if all(doc.get(k) == v for k, v in query.items())
        ])


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())


class FakeAdmin:
    def command(self, command):
        assert command == "ping"
        return {"ok": 1}


class FakeClient:
    def __init__(self):
        self.admin = FakeAdmin()
        self.databases = {}

    def __getitem__(self, name):
        return self.databases.setdefault(name, FakeDatabase())


class BrokenAdmin:
    def command(self, command):
        raise OSError("network unavailable")


class BrokenClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.admin = BrokenAdmin()


def config(country="fi", enabled=True):
    return StorageConfig(dataset="ep24", country=country, mongo_enabled=enabled)


def test_collection_prefix_generation_and_country_isolation():
    fi = config("fi")
    pl = config("pl")
    assert collection_prefix("ep24", "fi") == "laclaugpt_ep24_fi"
    assert fi.collection("memory") == "laclaugpt_ep24_fi_memory"
    assert fi.collection("rag") == "laclaugpt_ep24_fi_rag"
    assert pl.collection("memory") == "laclaugpt_ep24_pl_memory"
    assert pl.collection("rag") == "laclaugpt_ep24_pl_rag"
    assert fi.collection("analysis") != pl.collection("analysis")


def test_dataframe_mapping_and_export_round_trip():
    cfg = config()
    frame = pd.DataFrame([
        {"video_id": "abc", "text": "Hei", "score": 1},
        {"video_id": "def", "text": "Moi", "score": None},
    ])
    docs = dataframe_to_documents(
        frame, config=cfg, stage="summary", run_id="42",
        source_hint="ep24_fi.csv", model="gemma4:12b"
    )
    assert docs[0]["_storage_id"]
    assert docs[0]["_provenance"]["dataset"] == "ep24"
    assert docs[0]["_provenance"]["country"] == "fi"
    exported = documents_to_dataframe(docs)
    assert list(exported["video_id"]) == ["abc", "def"]
    assert "_storage_id" in exported.columns


def test_upsert_is_idempotent_and_memory_persists_across_instances():
    client = FakeClient()
    storage = Storage(MongoStorage(config(), client=client))
    storage.memory.put({"source_id": "doc-1", "text": "first", "memory_type": "entity"})
    storage.memory.put({"source_id": "doc-1", "text": "first", "memory_type": "entity"})
    assert len(storage.memory.search()) == 1
    second = Storage(MongoStorage(config(), client=client))
    assert second.memory.search()[0]["source_id"] == "doc-1"


def test_fi_and_pl_use_different_collections_on_same_database():
    client = FakeClient()
    fi = Storage(MongoStorage(config("fi"), client=client))
    pl = Storage(MongoStorage(config("pl"), client=client))
    fi.memory.put({"source_id": "same", "text": "Finland"})
    pl.memory.put({"source_id": "same", "text": "Poland"})
    assert fi.memory.search()[0]["text"] == "Finland"
    assert pl.memory.search()[0]["text"] == "Poland"


def test_rag_persistence_and_portable_retrieval():
    client = FakeClient()
    storage = Storage(MongoStorage(config(), client=client))
    storage.rag.upsert({
        "chunk_id": "c1",
        "source_id": "s1",
        "original_text": "European Parliament election in Finland",
        "translated_english": "European Parliament election in Finland",
        "language": "fi",
        "embedding_model": "example-v1",
        "embeddings": [0.1, 0.2],
    })
    result = storage.rag.retrieve("Parliament Finland")
    assert result[0]["chunk_id"] == "c1"


def test_dataframe_upsert_then_export_is_idempotent():
    client = FakeClient()
    mongo = MongoStorage(config(), client=client)
    frame = pd.DataFrame([
        {"id": "1", "summary_analysis": "x"},
        {"id": "2", "summary_analysis": "y"},
    ])
    assert mongo.dataframe_upsert("analysis", frame, stage="summary", run_id="r1") == 2
    assert mongo.dataframe_upsert("analysis", frame, stage="summary", run_id="r2") == 2
    exported = mongo.dataframe_export("analysis")
    assert len(exported) == 2
    assert set(exported["id"]) == {"1", "2"}


def test_disabled_mode_requires_no_mongodb(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "0")
    monkeypatch.setenv("LACLAUGPT_DATASET", "ep24")
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "fi")
    cfg = StorageConfig.from_env()
    assert cfg.mongo_enabled is False
    frame = pd.DataFrame([{"id": "1", "legacy_column": "preserved"}])
    assert frame.loc[0, "legacy_column"] == "preserved"


def test_connection_failure_is_clear():
    with pytest.raises(RuntimeError, match="external MongoDB connection failed"):
        MongoStorage(config(), client=BrokenClient())


def test_enabled_mode_requires_uri_without_injected_client():
    cfg = StorageConfig(dataset="ep24", country="fi", mongo_enabled=True, mongo_uri=None)
    with pytest.raises(RuntimeError, match="LACLAUGPT_MONGO_URI"):
        MongoStorage(cfg)


def test_reviewed_sqlite_memory_bridge_exports_only_canonical(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE objects(obj_id TEXT, kind TEXT, canonical_label TEXT, state TEXT, origin TEXT, updated_at TEXT)"
        )
        db.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?,?)",
            ("E-1", "entity", "Canonical actor", "CANONICAL", "researcher", "2026-10-01"),
        )
        db.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?,?)",
            ("E-2", "entity", "Proposal", "PROVISIONAL", "model", "2026-10-01"),
        )
    docs = sqlite_memory_documents(path, config=config())
    assert len(docs) == 1
    assert docs[0]["source_id"] == "E-1"
    assert docs[0]["memory_type"] == "entity"


def test_exported_storage_id_is_preserved_on_reimport():
    cfg = config()
    first = dataframe_to_documents(
        pd.DataFrame([{"id": "42", "text": "first"}]),
        config=cfg,
        stage="summary",
        run_id="a",
    )
    exported = documents_to_dataframe(first)
    second = dataframe_to_documents(
        exported,
        config=cfg,
        stage="summary",
        run_id="b",
    )
    assert second[0]["_storage_id"] == first[0]["_storage_id"]
