#!/usr/bin/env python3
"""Persistent EP24 research memory for CSC Roihu.

The default backend is SQLite and requires no service. Runtime jobs should use a
frozen accepted snapshot; discoveries are written as proposals and merged later
by a single writer. The legacy pipeline is never rewritten by this module.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KINDS = ("entity", "topic", "signifier", "target", "actor", "formation")
STATES = ("CANONICAL", "PROVISIONAL", "MERGED", "DEPRECATED", "REJECTED")
PREFIX = {"entity": "E", "topic": "T", "signifier": "S", "target": "C", "actor": "A", "formation": "F"}
SCHEMA_VERSION = 2


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
    candidates: tuple[dict[str, Any], ...] = ()


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

    def _schema_version(self) -> int:
        if not self.path.exists():
            return 0
        with self.connect() as db:
            pragma = int(db.execute("PRAGMA user_version").fetchone()[0])
            if pragma:
                return pragma
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            if exists:
                row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                if row and str(row[0]).isdigit():
                    return int(row[0])
        return 0

    def _backup_before_migration(self, version: int) -> Path:
        target = self.path.with_suffix(self.path.suffix + f".schema-v{version}.bak")
        tmp = target.with_suffix(target.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        # sqlite3.Connection context managers commit/rollback but do not close.
        # Close both handles before os.replace; Windows otherwise keeps tmp locked.
        with closing(self.connect()) as src, closing(sqlite3.connect(tmp)) as dst:
            src.backup(dst)
        tmp.replace(target)
        return target

    def _init(self) -> None:
        previous = self._schema_version()
        if previous > SCHEMA_VERSION:
            raise RuntimeError(
                f"memory schema {previous} is newer than supported {SCHEMA_VERSION}"
            )
        if 0 < previous < SCHEMA_VERSION:
            self._backup_before_migration(previous)

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
                    valid_from TEXT NOT NULL DEFAULT '', valid_to TEXT NOT NULL DEFAULT '',
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
                    UNIQUE(obj_id,source_type,source_ref,publication_date,evidence_locator),
                    FOREIGN KEY(obj_id) REFERENCES objects(obj_id)
                );
                CREATE TABLE IF NOT EXISTS affiliations(
                    affiliation_id INTEGER PRIMARY KEY AUTOINCREMENT, obj_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT '', organization_label TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '', valid_from TEXT NOT NULL DEFAULT '',
                    valid_to TEXT NOT NULL DEFAULT '', source_ref TEXT NOT NULL DEFAULT '',
                    UNIQUE(obj_id,organization_id,organization_label,role,valid_from,valid_to),
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
            columns = {row[1] for row in db.execute("PRAGMA table_info(objects)")}
            if "valid_from" not in columns:
                db.execute("ALTER TABLE objects ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''")
            if "valid_to" not in columns:
                db.execute("ALTER TABLE objects ADD COLUMN valid_to TEXT NOT NULL DEFAULT ''")
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def add_object(self, kind: str, label: str, *, obj_id: str | None = None, english_label: str = "", country: str = "", language: str = "", entity_type: str = "", disambiguation: str = "", definition: str = "", state: str = "PROVISIONAL", origin: str = "", locked: bool = False, valid_from: str = "", valid_to: str = "", preserve_upstream_id: str = "") -> str:
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
                    "UPDATE objects SET english_label=?,definition=?,entity_type=?,valid_from=?,valid_to=?,updated_at=? WHERE obj_id=?",
                    (english_label or existing["english_label"], definition or existing["definition"], entity_type or existing["entity_type"], valid_from or existing["valid_from"], valid_to or existing["valid_to"], ts, obj_id),
                )
            else:
                db.execute(
                    """INSERT INTO objects(
                        obj_id,kind,canonical_label,original_label,english_label,country,language,
                        entity_type,disambiguation,definition,state,origin,locked,valid_from,valid_to,
                        created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        obj_id, kind, canonical, canonical, english_label, country.upper(),
                        language.lower(), entity_type, disambiguation, definition, state, origin,
                        int(locked), valid_from, valid_to, ts, ts,
                    ),
                )
            self._add_alias_db(db, obj_id, canonical, country=country, language=language, provenance=origin)
            old_id = preserve_upstream_id or upstream_stable_id(kind, canonical)
            if old_id != obj_id:
                # An upstream id is derived from a label-only hash, so two genuinely
                # distinct objects can legitimately claim the same one (see
                # ``upstream_legacy_key``: accents and parenthesised qualifiers are
                # stripped). A second claimant is NOT dropped and NOT merged: the
                # row is written and the collision is recorded, so a later lookup can
                # report ambiguity instead of silently picking whichever object was
                # inserted last.
                prior = db.execute(
                    "SELECT DISTINCT ep24_id FROM id_crosswalk WHERE upstream_id=?",
                    (old_id,),
                ).fetchall()
                others = [row[0] for row in prior if row[0] != obj_id]
                db.execute("INSERT OR IGNORE INTO id_crosswalk VALUES(?,?,?)", (old_id, obj_id, "context-aware EP24 identity"))
                if others:
                    self._record_crosswalk_collision(db, old_id, obj_id, others)
                    # Record the existing claimants too: every object sharing an
                    # upstream id is part of the ambiguity, not just the last one
                    # to arrive, and an operator listing collisions needs all sides.
                    for other in others:
                        self._record_crosswalk_collision(
                            db, old_id, other, [obj_id] + [o for o in others if o != other]
                        )
        return obj_id

    def _record_crosswalk_collision(self, db: sqlite3.Connection, upstream_id: str,
                                    obj_id: str, others: list[str]) -> None:
        """Mark an upstream id that now maps to more than one object.

        Recorded once per (upstream_id, claimant) pair so re-running an idempotent
        import does not keep appending audit noise.
        """
        db.execute(
            """CREATE TABLE IF NOT EXISTS crosswalk_collisions(
                upstream_id TEXT NOT NULL,
                claimant_id TEXT NOT NULL,
                other_ids TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                PRIMARY KEY(upstream_id, claimant_id)
            )"""
        )
        db.execute(
            "INSERT OR IGNORE INTO crosswalk_collisions VALUES(?,?,?,?,?)",
            (
                upstream_id,
                obj_id,
                ",".join(sorted(others)),
                "label-only upstream id claimed by more than one object",
                now_iso(),
            ),
        )
        db.execute(
            "INSERT INTO audit_log(obj_id,action,actor,details_json,created_at) VALUES(?,?,?,?,?)",
            (
                obj_id, "crosswalk_collision", "system",
                json.dumps({"upstream_id": upstream_id, "other_ids": sorted(others)}),
                now_iso(),
            ),
        )

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
            rows = [r for r in rows if r["country"] in ("", country.upper())]
        if language:
            rows = [r for r in rows if r["language"] in ("", language.lower())]
        if len(rows) == 1:
            row = rows[0]
            return Resolution(raw, kind, "EXISTING", row["obj_id"], row["canonical_label"], "alias")
        if len(rows) > 1:
            return Resolution(raw, kind, "AMBIGUOUS", matched_via="alias")
        return Resolution(raw, kind, "NEW")

    def fuzzy_candidates(
        self,
        raw: str,
        kind: str,
        *,
        country: str = "",
        language: str = "",
        accepted_only: bool = True,
        limit: int = 3,
        threshold: float = 0.82,
    ) -> list[dict[str, Any]]:
        """Propose conservative fuzzy candidates; never resolves anything.

        Candidates are ranked by a plain ``difflib`` ratio over the same
        ``surface_key`` used for exact alias lookup, so a candidate can only
        differ from an exact alias hit by case/whitespace-insensitive spelling
        drift, never by the accent/qualifier stripping that caused the false
        merges fixed in known-issue #2. An exact-key match is excluded here
        because ``resolve()`` already finds it; this method is for the
        remaining, non-exact surface forms only. The caller decides whether a
        candidate is ever accepted: this never mutates memory state.
        """
        key = surface_key(raw)
        if not key:
            return []
        states = ("CANONICAL",) if accepted_only else ("CANONICAL", "PROVISIONAL")
        qs = ",".join("?" for _ in states)
        with self.connect() as db:
            rows = list(
                db.execute(
                    f"SELECT DISTINCT obj_id, canonical_label, english_label, country, language, state "
                    f"FROM objects WHERE kind=? AND state IN ({qs})",
                    [kind, *states],
                )
            )
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if country and row["country"] not in ("", country.upper()):
                continue
            if language and row["language"] not in ("", language.lower()):
                continue
            label_key = surface_key(row["canonical_label"])
            if not label_key or label_key == key:
                continue
            score = difflib.SequenceMatcher(None, key, label_key).ratio()
            if score >= threshold:
                candidates.append(
                    {
                        "obj_id": row["obj_id"],
                        "label": row["canonical_label"],
                        "english_label": row["english_label"],
                        "score": round(score, 4),
                        "match_method": "fuzzy_candidate",
                    }
                )
        candidates.sort(key=lambda c: (-c["score"], c["label"].casefold()))
        return candidates[: max(0, limit)]

    def resolve_identity(
        self,
        raw: str,
        kind: str,
        *,
        country: str = "",
        language: str = "",
        accepted_only: bool = True,
        fuzzy_threshold: float = 0.82,
        fuzzy_limit: int = 3,
    ) -> dict[str, Any]:
        """Exact-first, fuzzy-candidates-second identity resolution.

        Returns a plain, JSON-serializable dict so it can be embedded directly
        into additive enrichment columns. Decision order:

        1. ``EXISTING`` -- an exact canonical/alias match; safe to use.
        2. ``AMBIGUOUS`` -- more than one exact alias match; never resolved
           automatically, conservative fuzzy candidates are attached for a
           human to review but the decision itself still abstains.
        3. ``FUZZY_CANDIDATE`` -- no exact match, but one or more conservative
           fuzzy candidates were found; these are proposals only, never a
           silent canonization.
        4. ``NEW`` -- no exact or fuzzy match at all.
        """
        result = self.resolve(raw, kind, country=country, language=language, accepted_only=accepted_only)
        payload: dict[str, Any] = {
            "raw": raw,
            "kind": kind,
            "decision": result.decision,
            "obj_id": result.obj_id,
            "label": result.label,
            "match_method": result.matched_via or ("exact_alias" if result.decision == "EXISTING" else ""),
            "candidates": [],
        }
        if result.decision in ("NEW", "AMBIGUOUS"):
            candidates = self.fuzzy_candidates(
                raw, kind, country=country, language=language, accepted_only=accepted_only,
                limit=fuzzy_limit, threshold=fuzzy_threshold,
            )
            payload["candidates"] = candidates
            if candidates and result.decision == "NEW":
                payload["decision"] = "FUZZY_CANDIDATE"
                payload["match_method"] = "fuzzy_candidate"
        return payload

    def add_provenance(
        self,
        obj_id: str,
        *,
        source_type: str = "",
        source_ref: str = "",
        source_language: str = "",
        publication_date: str = "",
        event_valid_from: str = "",
        event_valid_to: str = "",
        retrieved_at: str = "",
        evidence_locator: str = "",
    ) -> None:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM objects WHERE obj_id=?", (obj_id,)).fetchone():
                raise KeyError(obj_id)
            db.execute(
                """INSERT OR IGNORE INTO provenance(
                    obj_id,source_type,source_ref,source_language,publication_date,
                    event_valid_from,event_valid_to,retrieved_at,evidence_locator
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    obj_id, source_type, source_ref, source_language, publication_date,
                    event_valid_from, event_valid_to, retrieved_at, evidence_locator,
                ),
            )

    def add_affiliation(
        self,
        obj_id: str,
        *,
        organization_id: str = "",
        organization_label: str = "",
        role: str = "",
        valid_from: str = "",
        valid_to: str = "",
        source_ref: str = "",
    ) -> None:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM objects WHERE obj_id=?", (obj_id,)).fetchone():
                raise KeyError(obj_id)
            db.execute(
                """INSERT OR IGNORE INTO affiliations(
                    obj_id,organization_id,organization_label,role,valid_from,valid_to,source_ref
                ) VALUES(?,?,?,?,?,?,?)""",
                (obj_id, organization_id, organization_label, role, valid_from, valid_to, source_ref),
            )

    def add_crosswalk(self, external_id: str, obj_id: str, *, reason: str) -> None:
        """Map an upstream/codebook identifier to the actual resolved EP24 object."""
        if not external_id:
            return
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM objects WHERE obj_id=?", (obj_id,)).fetchone():
                raise KeyError(obj_id)
            db.execute(
                "INSERT OR IGNORE INTO id_crosswalk(upstream_id,ep24_id,reason) VALUES(?,?,?)",
                (external_id, obj_id, reason),
            )

    def resolve_upstream_id(self, upstream_id: str) -> dict[str, Any]:
        """Resolve a legacy/upstream id to EP24 objects, reporting ambiguity.

        ``upstream_stable_id`` strips accents and parenthesised qualifiers, so a
        label-only upstream id is not guaranteed unique. Returning one object for
        an ambiguous id would silently pick a winner by insertion order, which is
        exactly what the issue forbids. This returns every claimant and marks the
        result ambiguous instead.

        An unknown id yields an empty list, not an error.
        """
        with self.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT ep24_id FROM id_crosswalk WHERE upstream_id=? ORDER BY ep24_id",
                (upstream_id,),
            ).fetchall()
        obj_ids = [row[0] for row in rows]
        return {
            "upstream_id": upstream_id,
            "obj_ids": obj_ids,
            "ambiguous": len(obj_ids) > 1,
        }

    def crosswalk_collisions(self) -> list[dict[str, Any]]:
        """Every recorded upstream-id collision, for audit and operator review."""
        with self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crosswalk_collisions'"
            ).fetchone()
            if not exists:
                return []
            rows = db.execute(
                "SELECT upstream_id, claimant_id, other_ids, reason, created_at "
                "FROM crosswalk_collisions ORDER BY upstream_id, claimant_id"
            ).fetchall()
        return [dict(row) for row in rows]

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

    def merge_proposal_shards(self, shard_paths: list[str | Path]) -> dict[str, int]:
        """Deterministically merge proposal-only SQLite shards into this memory DB."""
        inserted = 0
        duplicates = 0
        conflicts = 0
        proposal_columns = (
            "proposal_id", "kind", "raw_label", "country", "language", "proposed_obj_id",
            "reason", "source_record_id", "run_id", "stage", "model", "prompt_version",
            "payload_json", "status", "created_at",
        )
        collected: list[tuple[Any, ...]] = []
        for shard in sorted(Path(path) for path in shard_paths):
            if not shard.exists():
                raise FileNotFoundError(shard)
            with sqlite3.connect(shard) as source:
                source.row_factory = sqlite3.Row
                exists = source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='proposals'"
                ).fetchone()
                if not exists:
                    continue
                for row in source.execute("SELECT * FROM proposals ORDER BY proposal_id"):
                    collected.append(tuple(row[column] for column in proposal_columns))
        collected.sort(key=lambda row: (row[0], row[7], row[8], row[9]))

        with self.connect() as db:
            for values in collected:
                proposal_id = values[0]
                existing = db.execute(
                    "SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)
                ).fetchone()
                if existing is None:
                    db.execute(
                        "INSERT INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        values,
                    )
                    inserted += 1
                    continue
                existing_values = tuple(existing[column] for column in proposal_columns)
                if existing_values == values:
                    duplicates += 1
                else:
                    conflicts += 1
                    db.execute(
                        "INSERT INTO audit_log(action,obj_id,actor,details_json,created_at) VALUES(?,?,?,?,?)",
                        (
                            "proposal_conflict", "", "single-writer-merge",
                            json.dumps(
                                {
                                    "proposal_id": proposal_id,
                                    "kept": dict(zip(proposal_columns, existing_values, strict=True)),
                                    "rejected": dict(zip(proposal_columns, values, strict=True)),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            now_iso(),
                        ),
                    )
        return {"inserted": inserted, "duplicates": duplicates, "conflicts": conflicts}

    def snapshot(self, target: str | Path) -> Path:
        """Create a transactionally consistent SQLite snapshot via backup API."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        # sqlite3.Connection context managers commit/rollback but do not close.
        # Close both handles before os.replace; Windows otherwise keeps tmp locked.
        with closing(self.connect()) as src, closing(sqlite3.connect(tmp)) as dst:
            src.backup(dst)
        tmp.replace(target)
        return target

    def export_csv(self, directory: str | Path) -> list[Path]:
        import csv
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        outputs = []
        with self.connect() as db:
            for table in ("objects", "aliases", "redirects", "id_crosswalk", "provenance", "affiliations", "proposals", "audit_log"):
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
    merge = sub.add_parser("merge-proposals")
    merge.add_argument("shards", nargs="+")
    args = parser.parse_args(argv)
    memory = EP24Memory(args.db)
    if args.command == "snapshot":
        print(memory.snapshot(args.target))
    elif args.command == "merge-proposals":
        print(json.dumps(memory.merge_proposal_shards(args.shards), sort_keys=True))
    else:
        for path in memory.export_csv(args.directory):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
