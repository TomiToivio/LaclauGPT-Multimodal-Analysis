#!/usr/bin/env python3
"""Read-only EP24 FI/PL/PT export from shared MongoDB (#306).

    export LACLAUGPT_MONGODB_URI=...      # private environment, never echoed
    export LACLAUGPT_MONGODB_DATABASE=...
    python scripts/ep24/export_mongo_results.py \
        --country finland poland portugal \
        --output <private-export-dir> \
        --record-unit video

What it guarantees:

* **read-only** — only ``find`` calls on the client; no writes, no DDL, no indexes;
* **no invented identities** — rows without a usable id are quarantined and counted;
* **one canonical row per unit** — deterministic winner selection, so re-runs are byte-identical;
* **provenance in the manifest** — collections read, documents seen, pipeline markers, counts;
* **secrets stay out** — the URI and database are read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ep24_mongo_export as export  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only EP24 MongoDB export (#306).")
    parser.add_argument("--country", nargs="+", choices=export.COUNTRY_SLUGS,
                        default=list(export.COUNTRY_SLUGS))
    parser.add_argument("--output", required=True, type=Path, help="destination directory")
    parser.add_argument("--record-unit", choices=export.RECORD_UNITS,
                        default=export.DEFAULT_RECORD_UNIT,
                        help="video (one row per media, default) or ingest (one per pipeline run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="read at most N canonical rows per country (development subset)")
    parser.add_argument("--batch-size", type=int, default=1000, help="cursor batch size")
    parser.add_argument("--timeout-ms", type=int, default=15000, help="server selection timeout")
    parser.add_argument("--text-threshold", type=int, default=export.DEFAULT_TEXT_THRESHOLD,
                        help="strings longer than this go to the text bundle, not the row CSV")
    parser.add_argument("--children", nargs="*", choices=sorted(export.CHILD_SUFFIXES),
                        default=sorted(export.CHILD_SUFFIXES),
                        help="child collections to export alongside the canonical rows")
    return parser.parse_args(argv)


def iter_documents(collection: Any, *, batch_size: int, limit: int | None) -> Iterator[dict[str, Any]]:
    """Page through a collection deterministically (``_id`` order), read-only."""
    cursor = collection.find({}, batch_size=batch_size).sort("_id", 1)
    for index, document in enumerate(cursor):
        if limit is not None and index >= limit:
            break
        yield document


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        uri, database = export.config_from_env()
    except export.ExportError as error:
        raise SystemExit(str(error)) from error

    from pymongo import MongoClient  # imported late so the pure logic is testable without it

    output: Path = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    client = MongoClient(uri, serverSelectionTimeoutMS=args.timeout_ms, readPreference="primary")
    db = client[database]
    existing = set(db.list_collection_names())

    country_stats: dict[str, dict[str, Any]] = {}
    child_counts: dict[str, dict[str, int]] = {}
    collections_read: dict[str, dict[str, str]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    combined: list[dict[str, Any]] = []

    for country in args.country:
        source = export.collection_name(country, export.CANONICAL_SUFFIX)
        if source not in existing:
            raise SystemExit(f"collection missing for {country}: {source}")

        raw = list(iter_documents(db[source], batch_size=args.batch_size, limit=args.limit))
        kept, stats = export.select_canonical(raw, args.record_unit)
        projected = [export.project_row(row) for row in kept]

        # The lean frame answers "which records, with what coverage"; the bundle carries the prose.
        lean_rows: list[dict[str, Any]] = []
        heavy_records = 0
        for row in projected:
            lean, heavy = export.split_heavy_fields(row, threshold=args.text_threshold)
            lean_rows.append(lean)
            heavy_records += 1 if heavy else 0
        bundle = export.write_text_bundle(output / f"{country}_text.jsonl.gz", projected,
                                          threshold=args.text_threshold)
        stats["text_bundle_records"] = bundle
        rows = lean_rows

        written = export.write_rows(output / f"{country}.csv", rows)
        combined.extend(rows)
        country_stats[country] = {**stats, "rows_written": written,
                                  "source_documents": len(raw)}
        collections_read[country] = {"canonical": source}
        provenance[country] = export.discover_provenance(kept)

        child_counts[country] = {}
        for role in args.children:
            name = export.child_collection(country, role)
            if name not in existing:
                # Absence is a finding: Poland has no DNA statements collection, PL/PT have no SNA.
                child_counts[country][role] = 0
                collections_read[country][role] = "absent"
                continue
            documents = [export.project_row(doc) for doc in
                         iter_documents(db[name], batch_size=args.batch_size, limit=args.limit)]
            export.write_rows(output / "children" / f"{country}_{role}.csv", documents)
            child_counts[country][role] = len(documents)
            collections_read[country][role] = name

        print(f"{country}: {len(raw)} read -> {stats['records']} records "
              f"({stats['duplicates_collapsed']} collapsed, {stats['quarantined']} quarantined)")

    combined.sort(key=lambda row: str(row.get("record_id", "")))
    export.write_rows(output / "combined.csv", combined)

    outputs = ([output / "combined.csv"]
               + [output / f"{c}.csv" for c in args.country]
               + [output / f"{c}_text.jsonl.gz" for c in args.country])
    manifest = export.write_manifest(
        output,
        record_unit=args.record_unit,
        countries=country_stats,
        child_counts=child_counts,
        collections_read=collections_read,
        provenance=provenance,
        outputs=outputs,
    )
    print(f"manifest: {manifest['records']} records, {manifest['distinct_media']} distinct media, "
          f"unit={manifest['record_unit']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
