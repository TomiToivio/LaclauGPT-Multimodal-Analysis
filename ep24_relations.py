#!/usr/bin/env python3
"""Temporal, provenance-carrying EP24 relations for electoral lists and coalitions.

Background relations are context only. A mention of an electoral list resolves
to that list; related parties/candidates are never promoted into observed
evidence unless the current source mentions them.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RELATION_TYPES = {"has_member", "candidate_on", "member_of_list", "eu_group"}
BACKGROUND_EVIDENCE_ROLE = "background_relation_not_source_evidence"
OBSERVED_EVIDENCE_ROLE = "directly_observed_in_source"

@dataclass(frozen=True)
class Relation:
    source_id: str
    relation_type: str
    target_id: str
    country: str = ""
    election: str = ""
    valid_from: str = ""
    valid_to: str = ""
    source_ref: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"unsupported EP24 relation type: {self.relation_type}")
        if not self.source_id or not self.target_id:
            raise ValueError("source_id and target_id are required")

class EP24RelationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS relations(
                    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    country TEXT NOT NULL DEFAULT '',
                    election TEXT NOT NULL DEFAULT '',
                    valid_from TEXT NOT NULL DEFAULT '',
                    valid_to TEXT NOT NULL DEFAULT '',
                    source_ref TEXT NOT NULL DEFAULT '',
                    provenance TEXT NOT NULL DEFAULT '',
                    UNIQUE(source_id, relation_type, target_id, country, election, valid_from, valid_to, source_ref)
                );
                CREATE INDEX IF NOT EXISTS idx_rel_source
                    ON relations(source_id, relation_type, country, election);
                CREATE INDEX IF NOT EXISTS idx_rel_target
                    ON relations(target_id, relation_type, country, election);
            """)

    def add(self, relation: Relation) -> None:
        with self.connect() as db:
            db.execute("""
                INSERT OR IGNORE INTO relations(
                    source_id, relation_type, target_id, country, election,
                    valid_from, valid_to, source_ref, provenance
                ) VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                relation.source_id, relation.relation_type, relation.target_id,
                relation.country.upper(), relation.election, relation.valid_from,
                relation.valid_to, relation.source_ref, relation.provenance,
            ))

    def related(self, obj_id: str, *, country: str = "", election: str = "",
                relation_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(relation_types or RELATION_TYPES)
        if not wanted:
            return []
        unknown = wanted - RELATION_TYPES
        if unknown:
            raise ValueError(f"unsupported EP24 relation type(s): {sorted(unknown)}")
        placeholders = ",".join("?" for _ in wanted)
        params: list[Any] = [obj_id, *sorted(wanted)]
        sql = f"SELECT * FROM relations WHERE source_id=? AND relation_type IN ({placeholders})"
        if country:
            sql += " AND country IN ('', ?)"
            params.append(country.upper())
        if election:
            sql += " AND election IN ('', ?)"
            params.append(election)
        sql += " ORDER BY relation_type, target_id"
        with self.connect() as db:
            rows = list(db.execute(sql, params))
        return [dict(row) for row in rows]

def relation_context(observed_ids: Iterable[str], store: EP24RelationStore, *,
                     labels: dict[str, str] | None = None, country: str = "",
                     election: str = "") -> dict[str, Any]:
    labels = labels or {}
    seen: set[str] = set()
    observed: list[dict[str, str]] = []
    related: list[dict[str, Any]] = []
    for obj_id in observed_ids:
        obj_id = str(obj_id).strip()
        if not obj_id or obj_id in seen:
            continue
        seen.add(obj_id)
        observed.append({
            "obj_id": obj_id,
            "label": labels.get(obj_id, obj_id),
            "evidence_role": OBSERVED_EVIDENCE_ROLE,
        })
        for rel in store.related(obj_id, country=country, election=election):
            related.append({
                **rel,
                "source_label": labels.get(rel["source_id"], rel["source_id"]),
                "target_label": labels.get(rel["target_id"], rel["target_id"]),
                "evidence_role": BACKGROUND_EVIDENCE_ROLE,
            })
    return {
        "observed": observed,
        "related": related,
        "firewall": (
            "Related objects are background context only. They may disambiguate "
            "an observed list/person/party but must never be emitted as directly "
            "observed evidence unless independently present in the current source."
        ),
    }

def relation_context_json(*args: Any, **kwargs: Any) -> str:
    return json.dumps(relation_context(*args, **kwargs), ensure_ascii=False, sort_keys=True)

__all__ = [
    "BACKGROUND_EVIDENCE_ROLE", "OBSERVED_EVIDENCE_ROLE", "EP24RelationStore",
    "Relation", "relation_context", "relation_context_json",
]
