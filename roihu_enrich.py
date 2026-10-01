#!/usr/bin/env python3
"""Additive Memory/Codebook enrichment between postprocess and populism.

Disabled by default. When enabled it only appends columns to the existing EP24
CSV and never rewrites legacy entity/topic values. Country identity comes from
the explicit EP24 file manifest, not from language inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

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
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def seed_memory(private_root: Path, memory: EP24Memory) -> int:
    """Single-writer preparation: seed only reviewed/locked codebook objects."""
    seen: set[tuple[str, str]] = set()
    count = 0
    for country in COUNTRY_PROFILES:
        entries, _ = load_profile(private_root, country)
        for entry in entries:
            if entry.review_state != "CANONICAL" and not entry.locked:
                continue
            key = (entry.entry_id, country)
            if key in seen:
                continue
            seen.add(key)
            obj_id = memory.add_object(
                entry.kind,
                entry.label,
                english_label=entry.english_label,
                country="" if entry.country == "COMMON" else country,
                language=(entry.source_languages[0] if entry.source_languages else ""),
                entity_type=entry.entity_type,
                disambiguation=entry.disambiguation,
                definition=entry.english_definition or entry.definition,
                state="CANONICAL",
                origin=entry.origin,
                locked=entry.locked,
                preserve_upstream_id=entry.entry_id,
            )
            for alias in entry.aliases:
                memory.add_alias(obj_id, alias, country="" if entry.country == "COMMON" else country, provenance=entry.origin)
            count += 1
    return count


def enrich_file(path: Path, *, country: str, language: str, private_root: Path, memory: EP24Memory | None) -> dict:
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
