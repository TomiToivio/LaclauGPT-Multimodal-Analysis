#!/usr/bin/env python3
"""Additive Memory/Codebook enrichment between postprocess and populism.

Disabled by default. When enabled it only appends columns to the existing EP24
CSV and never rewrites legacy entity/topic values. Country identity comes from
the explicit EP24 file manifest, not from language inference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from roihu_codebooks import COUNTRY_PROFILES, context_block, load_profile
from roihu_memory import EP24Memory

EP24_FILES = {
    "FI": ("ep24_fi.csv", "fi"),
    "SE": ("ep24_sv.csv", "sv"),
    "PL": ("ep24_pl.csv", "pl"),
    "PT": ("ep24_pt.csv", "pt"),
    "DE": ("ep24_de.csv", "de"),
    "ES": ("ep24_es.csv", "es"),
    "HU": ("ep24_hu.csv", "hu"),
    "HR": ("ep24_hr.csv", "hr"),
    "FR": ("ep24_fr.csv", "fr"),
    "BG": ("ep24_bg.csv", "bg"),
}


def enabled() -> bool:
    return os.getenv("LACLAUGPT_ENRICHMENT_ENABLED", "0").casefold() in {"1", "true", "yes", "on"}


def split_values(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def seed_memory(private_root: Path, memory: EP24Memory) -> dict[str, int]:
    """Single-writer seeding without silently merging ambiguous aliases."""
    seen: set[tuple[str, str]] = set()
    counts = {"created": 0, "reused": 0, "ambiguous": 0, "alias_conflicts": 0}
    for country in COUNTRY_PROFILES:
        entries, _ = load_profile(private_root, country)
        for entry in entries:
            if entry.review_state != "CANONICAL" and not entry.locked:
                continue
            key = (entry.entry_id, country)
            if key in seen:
                continue
            seen.add(key)
            scope_country = "" if entry.country == "COMMON" else country
            scope_language = entry.source_languages[0] if entry.source_languages else ""
            existing = memory.resolve(
                entry.label,
                entry.kind,
                country=scope_country,
                language=scope_language,
                accepted_only=False,
            )
            if existing.decision == "AMBIGUOUS":
                memory.propose(
                    entry.kind,
                    entry.label,
                    country=scope_country,
                    language=scope_language,
                    reason="ambiguous-codebook-seed",
                    stage="memory-seed",
                    payload={"codebook_entry_id": entry.entry_id},
                )
                counts["ambiguous"] += 1
                continue
            if existing.decision == "EXISTING":
                obj_id = existing.obj_id
                memory.add_crosswalk(entry.entry_id, obj_id, reason="codebook entry resolved to existing object")
                counts["reused"] += 1
            else:
                obj_id = memory.add_object(
                    entry.kind,
                    entry.label,
                    english_label=entry.english_label,
                    country=scope_country,
                    language=scope_language,
                    entity_type=entry.entity_type,
                    disambiguation=entry.disambiguation,
                    definition=entry.english_definition or entry.definition,
                    state="CANONICAL",
                    valid_from=entry.valid_from,
                    valid_to=entry.valid_to,
                    origin=entry.origin,
                    locked=entry.locked,
                    preserve_upstream_id=entry.entry_id,
                )
                counts["created"] += 1

            for alias in entry.aliases:
                resolved_alias = memory.resolve(
                    alias,
                    entry.kind,
                    country=scope_country,
                    language=scope_language,
                    accepted_only=False,
                )
                if resolved_alias.decision == "NEW":
                    memory.add_alias(
                        obj_id,
                        alias,
                        country=scope_country,
                        language=scope_language,
                        provenance=entry.origin,
                    )
                elif resolved_alias.decision == "EXISTING" and resolved_alias.obj_id == obj_id:
                    continue
                else:
                    memory.propose(
                        entry.kind,
                        alias,
                        country=scope_country,
                        language=scope_language,
                        reason="codebook-alias-conflict",
                        stage="memory-seed",
                        payload={
                            "codebook_entry_id": entry.entry_id,
                            "target_obj_id": obj_id,
                            "resolved_obj_id": resolved_alias.obj_id,
                            "decision": resolved_alias.decision,
                        },
                    )
                    counts["alias_conflicts"] += 1

            for source in entry.sources:
                memory.add_provenance(
                    obj_id,
                    source_type=source.source_type,
                    source_ref=source.url or source.evidence_locator,
                    source_language=source.language,
                    publication_date=source.publication_date,
                    event_valid_from=entry.valid_from,
                    event_valid_to=entry.valid_to,
                    retrieved_at=source.retrieval_date,
                    evidence_locator=source.evidence_locator,
                )
            affiliations = entry.metadata.get("affiliations", [])
            if isinstance(affiliations, dict):
                affiliations = [affiliations]
            for affiliation in affiliations:
                if not isinstance(affiliation, dict):
                    continue
                memory.add_affiliation(
                    obj_id,
                    organization_id=str(affiliation.get("organization_id", "")),
                    organization_label=str(affiliation.get("organization_label", affiliation.get("organization", ""))),
                    role=str(affiliation.get("role", "")),
                    valid_from=str(affiliation.get("valid_from", "")),
                    valid_to=str(affiliation.get("valid_to", "")),
                    source_ref=str(affiliation.get("source_ref", "")),
                )
    return counts


def enrich_file(path: Path, *, country: str, language: str, private_root: Path, memory: EP24Memory | None) -> dict:
    import pandas as pd

    entries, profile = load_profile(private_root, country, language=language)
    frame = pd.read_csv(path)
    before_columns = list(frame.columns)
    new_columns = (
        "ep24_codebook_fingerprint",
        "ep24_codebook_context_json",
        "ep24_memory_entity_ids",
        "ep24_memory_topic_ids",
        "ep24_memory_unresolved_json",
    )
    for column in new_columns:
        if column not in frame.columns:
            frame[column] = ""

    for index, row in frame.iterrows():
        text = str(row.get("summary_analysis") or "")
        _block, selection = context_block(text, entries, country=country, language=language)
        frame.at[index, "ep24_codebook_fingerprint"] = profile["fingerprint"]
        frame.at[index, "ep24_codebook_context_json"] = json.dumps(selection, ensure_ascii=False, sort_keys=True)

        entity_ids: list[str] = []
        topic_ids: list[str] = []
        unresolved: list[dict[str, str]] = []
        if memory is not None:
            for kind, column, output in (("entity", "entities", entity_ids), ("topic", "topics", topic_ids)):
                for label in split_values(row.get(column)):
                    result = memory.resolve(label, kind, country=country, language=language, accepted_only=True)
                    if result.decision == "EXISTING":
                        output.append(result.obj_id)
                    else:
                        unresolved.append({"kind": kind, "raw": label, "decision": result.decision})
        frame.at[index, "ep24_memory_entity_ids"] = json.dumps(entity_ids, ensure_ascii=False)
        frame.at[index, "ep24_memory_topic_ids"] = json.dumps(topic_ids, ensure_ascii=False)
        frame.at[index, "ep24_memory_unresolved_json"] = json.dumps(unresolved, ensure_ascii=False, sort_keys=True)

    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "country": country,
        "language": language,
        "rows": len(frame),
        "legacy_columns_preserved": before_columns == [c for c in frame.columns if c in before_columns],
        "profile": profile,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", default=os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "."))
    parser.add_argument("--memory-db", default=os.getenv("LACLAUGPT_MEMORY_DB", "./database/ep24_memory.sqlite3"))
    parser.add_argument("--prepare-memory", action="store_true")
    parser.add_argument("--force", action="store_true", help="run even when LACLAUGPT_ENRICHMENT_ENABLED is false")
    args = parser.parse_args(argv)
    root = Path(args.private_root)
    memory = EP24Memory(args.memory_db)
    if args.prepare_memory:
        print(json.dumps({"seeded": seed_memory(root, memory), "memory_db": str(memory.path)}, indent=2))
        return 0
    if not (enabled() or args.force):
        print(json.dumps({"enabled": False, "status": "no-op", "reason": "set LACLAUGPT_ENRICHMENT_ENABLED=1 to append Memory/Codebook fields"}))
        return 0
    reports = []
    for country, (filename, language) in EP24_FILES.items():
        path = Path(filename)
        if path.exists():
            reports.append(enrich_file(path, country=country, language=language, private_root=root, memory=memory))
    print(json.dumps({"enabled": True, "reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
