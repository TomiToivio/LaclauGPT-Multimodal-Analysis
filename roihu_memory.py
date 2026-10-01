#!/usr/bin/env python3
"""Persistent EP24 research memory for CSC Roihu.

The default backend is SQLite and requires no service. Runtime jobs should use a
frozen accepted snapshot; discoveries are written as proposals and merged later
by a single writer. The legacy pipeline is never rewritten by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KINDS = ("entity", "topic", "signifier", "target", "actor", "formation")
STATES = ("CANONICAL", "PROVISIONAL", "MERGED", "DEPRECATED", "REJECTED")
PREFIX = {"entity": "E", "topic": "T", "signifier": "S", "target": "C", "actor": "A", "formation": "F"}
SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def surface_key(text: str) -> str:
    """Conservative lookup key: Unicode NFC + casefold + whitespace only."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).strip().casefold().split())


def upstream_legacy_key(text: str) -> str:
    """Compatibility key matching the older accent/qualifier-stripping ID logic."""
    value = unicodedata.normalize("NFD", str(text or "").strip()).casefold()
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = " ".join(value.split())
    if value.endswith(")") and "(" in value:
        value = value[: value.rfind("(")].strip()
    return value.strip(" \t\n.,;:!?\"'()[]{}")


def stable_id(kind: str, label: str, *, country: str = "", language: str = "", disambiguation: str = "") -> str:
    if kind not in KINDS:
        raise ValueError(f"unsupported memory kind: {kind}")
    raw = "|".join((kind, surface_key(label), country.upper().strip(), language.lower().strip(), surface_key(disambiguation)))
    if not surface_key(label):
        raise ValueError("memory label may not be empty")
    return f"{PREFIX[kind]}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def upstream_stable_id(kind: str, label: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unsupported memory kind: {kind}")
    key = upstream_legacy_key(label)
    if not key:
        raise ValueError("memory label may not be empty")
    return f"{PREFIX[kind]}-{hashlib.sha256(f'{kind}:{key}'.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class Resolution:
    raw: str
    kind: str
    decision: str
    obj_id: str = ""
    label: str = ""
    matched_via: str = ""


class EP24Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS objects(
                    obj_id TEXT PRIMARY KEY, kind TEXT NOT NULL, canonical_label TEXT NOT NULL,
                    original_label TEXT NOT NULL, english_label TEXT NOT NULL DEFAULT '',
                    country TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT '',
                    entity_type TEXT NOT NULL DEFAULT '', disambiguation TEXT NOT NULL DEFAULT '',
                    definition TEXT NOT NULL DEFAULT '', state TEXT NOT NULL DEFAULT 'PROVISIONAL',
                    origin TEXT NOT NULL DEFAULT '', locked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS aliases(
                    alias_id INTEGER PRIMARY KEY AUTOINCREMENT, obj_id TEXT NOT NULL,
                    raw_label TEXT NOT NULL, lookup_key TEXT NOT NULL, country TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT '',
                    UNIQUE(obj_id, raw_label, country, language), FOREIGN KEY(obj_id) REFERENCES objects(obj_id)
                );
                CREATE INDEX IF NOT EXISTS idx_alias_lookup ON aliases(lookup_key, country, language);
                CREATE TABLE IF NOT EXISTS redirects(old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS id_crosswalk(upstream_id TEXT NOT NULL, ep24_id TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(upstream_id, ep24_id));
                CREATE TABLE IF NOT EXISTS provenance(
                    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT, obj_id TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT '', source_ref TEXT NOT NULL DEFAULT '',
                    source_language TEXT NOT NULL DEFAULT '', publication_date TEXT NOT NULL DEFAULT '',
                    event_valid_from TEXT NOT NULL DEFAULT '', event_valid_to TEXT NOT NULL DEFAULT '',
                    retrieved_at TEXT NOT NULL DEFAULT '', evidence_locator TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(obj_id) REFERENCES objects(obj_id)
                );
                CREATE TABLE IF NOT EXISTS proposals(
                    proposal_id TEXT PRIMARY KEY, kind TEXT NOT NULL, raw_label TEXT NOT NULL,
                    country TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT '',
                    proposed_obj_id TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL,
                    source_record_id TEXT NOT NULL DEFAULT '', run_id TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '', prompt_version TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'OPEN', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log(
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, obj_id TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT 'researcher', details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                """
            )
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    def add_object(self, kind: str, label: str, *, obj_id: str | None = None, english_label: str = "", country: str = "", language: str = "", entity_type: str = "", disambiguation: str = "", definition: str = "", state: str = "PROVISIONAL", origin: str = "", locked: bool = False, preserve_upstream_id: str = "") -> str:
        if kind not in KINDS or state not in STATES:
            raise ValueError("unsupported kind/state")
        canonical = str(label).strip()
        if not canonical:
            raise ValueError("memory label may not be empty")
        obj_id = obj_id or stable_id(kind, canonical, country=country, language=language, disambiguation=disambiguation)
        ts = now_iso()
        with self.connect() as db:
            existing = db.execute("SELECT * FROM objects WHERE obj_id=?", (obj_id,)).fetchone()
            if existing:
                if int(existing["locked"]):
                    return obj_id
                db.execute(
                    "UPDATE objects SET english_label=?,definition=?,entity_type=?,updated_at=? WHERE obj_id=?",
                    (english_label or existing["english_label"], definition or existing["definition"], entity_type or existing["entity_type"], ts, obj_id),
                )
            else:
                db.execute(
                    "INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (obj_id, kind, canonical, canonical, english_label, country.upper(), language.lower(), entity_type, disambiguation, definition, state, origin, int(locked), ts, ts),
                )
            self._add_alias_db(db, obj_id, canonical, country=country, language=language, provenance=origin)
            old_id = preserve_upstream_id or upstream_stable_id(kind, canonical)
            if old_id != obj_id:
                db.execute("INSERT OR IGNORE INTO id_crosswalk VALUES(?,?,?)", (old_id, obj_id, "context-aware EP24 identity"))
        return obj_id

    def _add_alias_db(self, db: sqlite3.Connection, obj_id: str, alias: str, *, country: str = "", language: str = "", provenance: str = "") -> None:
        key = surface_key(alias)
        if not key:
            raise ValueError("memory alias may not be empty")
        db.execute(
            "INSERT OR IGNORE INTO aliases(obj_id,raw_label,lookup_key,country,language,provenance) VALUES(?,?,?,?,?,?)",
            (obj_id, unicodedata.normalize("NFC", alias.strip()), key, country.upper(), language.lower(), provenance),
        )

    def add_alias(self, obj_id: str, alias: str, *, country: str = "", language: str = "", provenance: str = "") -> None:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM objects WHERE obj_id=?", (obj_id,)).fetchone():
                raise KeyError(obj_id)
            self._add_alias_db(db, obj_id, alias, country=country, language=language, provenance=provenance)

    def resolve(self, raw: str, kind: str, *, country: str = "", language: str = "", accepted_only: bool = True) -> Resolution:
        key = surface_key(raw)
        if not key:
            return Resolution(raw, kind, "NEW")
        states = ("CANONICAL",) if accepted_only else ("CANONICAL", "PROVISIONAL", "DEPRECATED")
        qs = ",".join("?" for _ in states)
        params: list[Any] = [key, kind, *states]
        sql = f"SELECT DISTINCT o.* FROM aliases a JOIN objects o ON o.obj_id=a.obj_id WHERE a.lookup_key=? AND o.kind=? AND o.state IN ({qs})"
        with self.connect() as db:
            rows = list(db.execute(sql, params))
        if country:
            scoped = [r for r in rows if r["country"] in ("", country.upper())]
            rows = scoped or rows
        if language:
            scoped = [r for r in rows if r["language"] in ("", language.lower())]
            rows = scoped or rows
        if len(rows) == 1:
            row = rows[0]
            return Resolution(raw, kind, "EXISTING", row["obj_id"], row["canonical_label"], "alias")
        if len(rows) > 1:
            return Resolution(raw, kind, "AMBIGUOUS", matched_via="alias")
        return Resolution(raw, kind, "NEW")

    def set_state(self, obj_id: str, state: str, *, actor: str = "researcher") -> None:
        if state not in STATES:
            raise ValueError(state)
        with self.connect() as db:
            row = db.execute("SELECT locked FROM objects WHERE obj_id=?", (obj_id,)).fetchone()
            if not row:
                raise KeyError(obj_id)
            db.execute("UPDATE objects SET state=?,updated_at=? WHERE obj_id=?", (state, now_iso(), obj_id))
            db.execute("INSERT INTO audit_log(action,obj_id,actor,details_json,created_at) VALUES(?,?,?,?,?)", ("set_state", obj_id, actor, json.dumps({"state": state}), now_iso()))

    def propose(self, kind: str, label: str, *, country: str = "", language: str = "", reason: str, source_record_id: str = "", run_id: str = "", stage: str = "", model: str = "", prompt_version: str = "", payload: dict[str, Any] | None = None) -> str:
        proposal_id = "P-" + hashlib.sha256("|".join((kind, surface_key(label), country, language, source_record_id, stage)).encode()).hexdigest()[:20]
        proposed = stable_id(kind, label, country=country, language=language)
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (proposal_id, kind, label, country.upper(), language.lower(), proposed, reason, source_record_id, run_id, stage, model, prompt_version, json.dumps(payload or {}, ensure_ascii=False, sort_keys=True), "OPEN", now_iso()),
            )
        return proposal_id

    def redirect(self, old_id: str, new_id: str, *, reason: str, actor: str = "researcher") -> None:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM objects WHERE obj_id=?", (new_id,)).fetchone():
                raise KeyError(new_id)
            db.execute("INSERT OR REPLACE INTO redirects VALUES(?,?,?,?)", (old_id, new_id, reason, now_iso()))
            db.execute("INSERT INTO audit_log(action,obj_id,actor,details_json,created_at) VALUES(?,?,?,?,?)", ("redirect", new_id, actor, json.dumps({"old_id": old_id, "reason": reason}), now_iso()))

    def snapshot(self, target: str | Path) -> Path:
        """Create a transactionally consistent SQLite snapshot via backup API."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with self.connect() as src, sqlite3.connect(tmp) as dst:
            src.backup(dst)
        tmp.replace(target)
        return target

    def export_csv(self, directory: str | Path) -> list[Path]:
        import csv
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        outputs = []
        with self.connect() as db:
            for table in ("objects", "aliases", "redirects", "id_crosswalk", "proposals", "audit_log"):
                rows = list(db.execute(f"SELECT * FROM {table}"))
                path = directory / f"memory_{table}.csv"
                with path.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh)
                    if rows:
                        writer.writerow(rows[0].keys())
                        writer.writerows([tuple(row) for row in rows])
                outputs.append(path)
        return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./database/ep24_memory.sqlite3")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("target")
    exp = sub.add_parser("export-csv")
    exp.add_argument("directory")
    args = parser.parse_args(argv)
    memory = EP24Memory(args.db)
    if args.command == "snapshot":
        print(memory.snapshot(args.target))
    else:
        for path in memory.export_csv(args.directory):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
