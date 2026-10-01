"""EP24 pipeline persistence: MongoDB as canonical state, CSV/SQLite as backup (#64).

MongoDB is the primary durable store for the rerun; Redis is coordination/cache
and SQLite is a local checkpoint/back-up. Those three roles must not blur, so this
module owns only the Mongo side (and the namespacing that keeps EP24 collections
from colliding with other LaclauGPT projects).

Two things are deliberate:

- **Import safety.** `pymongo` is an optional extra, not a hard dependency: it is
  imported lazily, so the pipeline modules and their tests still import on a CI
  machine that has never installed it. Callers get a clear error naming the extra
  rather than a bare ImportError at collection time -- the failure mode that
  turned a missing `rdflib` into a whole-suite collection error in this repo.
- **One place that names collections.** `COLLECTIONS` is the contract; nothing
  else hard-codes a collection name, so the separation the issue asks for (source
  records, stage outputs, state, memory, RAG, codebooks, entities, themes, DNA,
  SNA, provenance, dead-letter) stays inspectable.

Nothing here reads or writes private research data by itself; it is storage
plumbing, and the callers decide what goes in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ep24_stage_state import Record, StateStore

#: Collection namespaces. EP24 is namespaced so it cannot collide with AI26 or
#: the other LaclauGPT projects sharing the same MongoDB deployment.
COLLECTIONS: dict[str, str] = {
    "records": "ep24_records",            # canonical source/dataframe records
    "stage_outputs": "ep24_stage_outputs",
    "stage_state": "ep24_stage_state",    # pipeline state/checkpoints
    "runs": "ep24_runs",                  # run provenance
    "errors": "ep24_dead_letter",         # errors/dead-letter/retry records
    "memory": "ep24_memory",
    "rag_documents": "ep24_rag_documents",
    "rag_embeddings": "ep24_rag_embeddings",
    "codebooks": "ep24_codebooks",
    "entities": "ep24_entities",
    "entity_aliases": "ep24_entity_aliases",
    "themes": "ep24_themes",
    "theme_aliases": "ep24_theme_aliases",
    "dna": "ep24_dna",
    "sna": "ep24_sna",
}

DEFAULT_DATABASE = "laclaugpt_ep24"

#: The environment variable the pipeline reads the URI from. Never a literal.
MONGO_URI_ENV = "LACLAUGPT_MONGO_URI"
MONGO_DB_ENV = "LACLAUGPT_EP24_DB"


def mongo_uri() -> str:
    """Read the MongoDB URI from the environment.

    Deliberately not defaulted: a missing URI must fail loudly here rather than
    silently connecting somewhere unintended.
    """
    uri = os.getenv(MONGO_URI_ENV, "").strip()
    if not uri:
        raise RuntimeError(
            f"{MONGO_URI_ENV} is not set; export it (or load the private env file) "
            "before running a MongoDB-backed stage. Never commit the URI."
        )
    return uri


def database_name() -> str:
    return os.getenv(MONGO_DB_ENV, DEFAULT_DATABASE).strip() or DEFAULT_DATABASE


def connect(*, uri: str | None = None, database: str | None = None):
    """Return (client, database) connected to the EP24 database."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - only on a machine without the extra
        raise RuntimeError(
            "pymongo is required for MongoDB-backed stages; install the project's "
            "'mongo' extra (pymongo>=4.10,<5)."
        ) from exc

    client = MongoClient(uri or mongo_uri())
    return client, client[database or database_name()]


def ensure_indexes(db) -> list[str]:
    """Create the indexes the stages rely on. Idempotent.

    Returns the collections it touched so a caller can log them.
    """
    touched: list[str] = []

    db[COLLECTIONS["records"]].create_index("record_id", unique=True)
    db[COLLECTIONS["records"]].create_index([("country", 1), ("source_row_index", 1)])

    state = db[COLLECTIONS["stage_state"]]
    state.create_index([("record_id", 1), ("stage", 1)], unique=True)
    # Finding claimable work by (stage, state) is the hot query for every stage.
    state.create_index([("stage", 1), ("state", 1)])
    # Stale-claim recovery scans by claim age.
    state.create_index([("stage", 1), ("claimed_at", 1)])

    db[COLLECTIONS["stage_outputs"]].create_index([("record_id", 1), ("stage", 1)], unique=True)
    db[COLLECTIONS["rag_documents"]].create_index("doc_id")
    db[COLLECTIONS["entities"]].create_index("canonical_id")
    db[COLLECTIONS["themes"]].create_index("canonical_id")
    db[COLLECTIONS["errors"]].create_index([("stage", 1), ("record_id", 1)])

    touched.extend(
        [
            COLLECTIONS["records"],
            COLLECTIONS["stage_state"],
            COLLECTIONS["stage_outputs"],
            COLLECTIONS["rag_documents"],
            COLLECTIONS["entities"],
            COLLECTIONS["themes"],
            COLLECTIONS["errors"],
        ]
    )
    return touched


def record_id_for(country: str, video_id: str, source_row_index: int) -> str:
    """Stable record identity (issue #64).

    The issue warns that repeated historical `video_id` values must not overwrite
    one another, so the id combines country, video and the source row index --
    not the video id alone. Readable rather than a hash, because these ids appear
    in logs, RDF and CSV backups and a human has to be able to trace them.
    """
    return f"{country}|{video_id}|{source_row_index}"


def upsert_records(db, records: list[dict]) -> int:
    """Idempotently write canonical records. Returns the number written."""
    from pymongo import UpdateOne

    if not records:
        return 0
    operations = []
    for record in records:
        if "record_id" not in record:
            raise ValueError("every record needs a record_id")
        operations.append(
            UpdateOne({"record_id": record["record_id"]}, {"$set": record}, upsert=True)
        )
    result = db[COLLECTIONS["records"]].bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


class MongoStateStore(StateStore):
    """Mongo-backed implementation of the stage-state store protocol.

    Kept thin on purpose: the eligibility/claim rules live in
    ``ep24_stage_state`` and are shared with the in-memory store, so the two
    cannot drift.
    """

    def __init__(self, db, *, stage: int) -> None:
        self.db = db
        self.stage = stage
        self.collection = db[COLLECTIONS["stage_state"]]

    def load(self, stage: int) -> dict[str, Record]:
        out: dict[str, Record] = {}
        for doc in self.collection.find({"stage": stage}):
            out[doc["record_id"]] = Record(
                record_id=doc["record_id"],
                stage=doc.get("stage", stage),
                state=doc.get("state", "pending"),
                attempts=int(doc.get("attempts", 0)),
                claimed_at=doc.get("claimed_at"),
                claimed_by=doc.get("claimed_by"),
                completed_at=doc.get("completed_at"),
                error=doc.get("error", ""),
                provenance=doc.get("provenance", {}) or {},
            )
        return out

    def save(self, record: Record) -> None:
        self.collection.update_one(
            {"record_id": record.record_id, "stage": record.stage},
            {
                "$set": {
                    "record_id": record.record_id,
                    "stage": record.stage,
                    "state": record.state,
                    "attempts": record.attempts,
                    "claimed_at": record.claimed_at,
                    "claimed_by": record.claimed_by,
                    "completed_at": record.completed_at,
                    "error": record.error,
                    "provenance": record.provenance,
                }
            },
            upsert=True,
        )


@dataclass
class RunProvenance:
    """What produced a stage's output, recorded once per run (issue #64).

    Kept as a plain dataclass with a `to_doc` so it can be written to Mongo and
    also embedded in a CSV backup manifest without a second code path.
    """

    run_id: str
    stage: int
    country: str
    started_at: str
    model: str = ""
    prompt_version: str = ""
    prompt_sha256: str = ""
    code_version: str = ""
    codebook_fingerprint: str = ""
    hostname: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "country": self.country,
            "started_at": self.started_at,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "code_version": self.code_version,
            "codebook_fingerprint": self.codebook_fingerprint,
            "hostname": self.hostname,
        }
