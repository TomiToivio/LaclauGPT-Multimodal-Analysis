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
import gc
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ep24_mongo_export as export  # noqa: E402


def _merge_contracts(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fold per-country contract blocks into the single mapping ``write_manifest`` merges."""
    counts: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for country, block in parts.items():
        counts.update(block["record_counts"])
        coverage.update(block["stage_coverage"])
        inputs.update(block["inputs"])
    counts.pop("exported_total", None)
    counts["exported_total"] = sum(int(entry.get("exported", 0)) for entry in counts.values()
                                   if isinstance(entry, dict))
    return {"record_counts": counts, "stage_coverage": coverage, "inputs": inputs}


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


#: Errors that mean "the server dropped this cursor", not "the data is wrong". A full-corpus export
#: reads for many minutes; the server closes idle cursors and the client must resume rather than
#: abandon a half-written country.
_RESUMABLE = ("_OperationCancelled", "CursorNotFound", "AutoReconnect", "NetworkTimeout",
              "ExecutionTimeout", "ServerSelectionTimeout", "ConnectionFailure",
              "NotPrimaryError", "OperationFailure")


def _is_resumable(error: BaseException) -> bool:
    return any(name in type(error).__name__ for name in _RESUMABLE)


def iter_documents(collection: Any, *, batch_size: int, limit: int | None,
                   retries: int = 6) -> Iterator[dict[str, Any]]:
    """Page through a collection deterministically (``_id`` order), read-only.

    A dropped cursor resumes from the last ``_id`` handed out, so a long export survives the server
    closing an idle cursor instead of dying after the first country has already been written.
    """
    last_id: Any = None
    delivered = 0
    attempt = 0
    while True:
        query: dict[str, Any] = {} if last_id is None else {"_id": {"$gt": last_id}}
        try:
            cursor = collection.find(query, batch_size=batch_size).sort("_id", 1)
            cursor.no_cursor_timeout = True
            for document in cursor:
                if limit is not None and delivered >= limit:
                    return
                last_id = document.get("_id", last_id)
                delivered += 1
                yield document
            return
        except Exception as error:  # noqa: BLE001 - narrowed by _is_resumable below
            if not _is_resumable(error) or attempt >= retries:
                raise
            attempt += 1
            # Back off up to 30 s: a dropped cursor or a reset connection is a pause, not a failure,
            # and the resumed query starts from the last id handed out.
            time.sleep(min(2 * attempt, 30))


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
    contract_parts: dict[str, dict[str, Any]] = {}
    child_counts: dict[str, dict[str, int]] = {}
    qa_parts: dict[str, dict] = {}
    heavy_report: dict[str, dict[str, int]] = {}
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

        heavy_report[country] = export.heavy_field_report(projected, threshold=args.text_threshold)
        written = export.write_rows(output / f"{country}.csv", rows)
        combined.extend(rows)
        read_count = len(raw)
        country_stats[country] = {**stats, "rows_written": written,
                                  "source_documents": read_count}
        collections_read[country] = {"canonical": source}
        provenance[country] = export.discover_provenance(kept)
        # Coverage is measured on the projected rows, before heavy text moves to the bundle: a step
        # whose only output is a long field is still a step that ran. Only the resulting counts are
        # kept, so peak memory stays one country rather than the whole corpus.
        # Children are read while the country's frames are still in memory: coverage for a step whose
        # output is a child table must be computed with the child counts, or it reports "unavailable"
        # for a stage that produced thousands of statements (#332).
        child_counts[country] = {}
        child_rows[country] = {}
        for role in args.children:
            name = export.child_collection(country, role)
            if name not in existing:
                # Absence is a finding: Poland has no DNA statements collection, PL/PT have no SNA.
                child_counts[country][role] = 0
                collections_read[country][role] = "absent"
                continue
            documents = [export.project_row(doc) for doc in
                         iter_documents(db[name], batch_size=args.batch_size, limit=args.limit)]
            # An explicit, documented parent key: the alias is derived from what the pipeline wrote,
            # never synthesised. Without it consumers join on `_storage_id` and silently get nothing.
            for document in documents:
                source_parent = str(document.get("source_record_id") or "").strip()
                if source_parent and not document.get("parent_record_id"):
                    document["parent_record_id"] = source_parent
            export.write_rows(output / "children" / f"{country}_{role}.csv", documents)
            child_counts[country][role] = len(documents)
            child_rows[country][role] = documents
            collections_read[country][role] = name

        contract_parts[country] = export.dashboard_contract(
            {country: {**stats, "source_documents": read_count}}, {country: projected},
            child_counts={country: child_counts[country]})
        # QA the child join while both sides are in memory: the parent key is `source_record_id` on the
        # child and `_storage_id` on the parent, which is documented but was not checkable before.
        parent_ids = {str(row.get("_storage_id") or "") for row in projected} | {
            str(row.get("record_id") or "") for row in projected}
        for role, documents in child_rows.get(country, {}).items():
            if documents:
                qa_parts[f"{country}:{role}"] = export.children_qa(documents, parent_ids)

        # Release this country's frames before the next one is read: a full corpus otherwise sits at
        # the memory ceiling by Portugal and dies while hashing the last bundle. The child rows are
        # kept only long enough to QA the parent join, then dropped with them.
        child_rows[country] = {}
        del projected, lean_rows, raw, kept
        gc.collect()

        print(f"{country}: {read_count} read -> {stats['records']} records "
              f"({stats['duplicates_collapsed']} collapsed, {stats['quarantined']} quarantined)")

    combined.sort(key=lambda row: str(row.get("record_id", "")))
    export.write_rows(output / "combined.csv", combined)

    outputs = ([output / "combined.csv"]
               + [output / f"{c}.csv" for c in args.country]
               + [output / f"{c}_text.jsonl.gz" for c in args.country])
    qa_report = qa_parts
    manifest = export.write_manifest(
        output,
        record_unit=args.record_unit,
        children_unit="ingest",
        heavy_fields=heavy_report,
        children_qa_report=qa_report,
        countries=country_stats,
        child_counts=child_counts,
        collections_read=collections_read,
        provenance=provenance,
        outputs=outputs,
        contract=_merge_contracts(contract_parts),
    )
    print(f"manifest: {manifest['records']} records, {manifest['distinct_media']} distinct media, "
          f"unit={manifest['record_unit']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
