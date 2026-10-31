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
    child_outputs: list[Path] = []
    child_counts: dict[str, dict[str, int]] = {}
    child_detail: dict[str, dict[str, dict[str, int]]] = {}
    children_contract: dict[str, dict[str, Any]] = {}
    child_statement_countries: dict[str, set[str]] = {}
    heavy_fields: dict[str, dict[str, int]] = {}
    heavy_splits: dict[str, dict[str, Any]] = {}
    collections_read: dict[str, dict[str, str]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    combined: list[dict[str, Any]] = []

    for country in dict.fromkeys(args.country):
        source = export.collection_name(country, export.CANONICAL_SUFFIX)
        if source not in existing:
            raise SystemExit(f"collection missing for {country}: {source}")

        raw = list(iter_documents(db[source], batch_size=args.batch_size, limit=args.limit))
        kept, stats = export.select_canonical(raw, args.record_unit)
        projected = [export.project_row(row) for row in kept]
        # Every row carries its own country. Without this the per-country CSV has no country column and the
        # child join (parent_country -> parent_record_id) has nothing to join on.
        for row in projected:
            row["country"] = country

        # The lean frame answers "which records, with what coverage"; the bundle carries the prose.
        lean_rows: list[dict[str, Any]] = []
        heavy_records = 0
        heavy_fields[country] = {}
        for row in projected:
            lean, heavy = export.split_heavy_fields(row, threshold=args.text_threshold)
            lean_rows.append(lean)
            heavy_records += 1 if heavy else 0
            # Which cells left the CSV, and how many rows each: elision is data, not a silent hole.
            for name in heavy:
                heavy_fields[country][name] = heavy_fields[country].get(name, 0) + 1
        heavy_splits[country] = {"threshold": args.text_threshold,
                                 "records_with_heavy_fields": heavy_records,
                                 "fields": dict(sorted(heavy_fields[country].items()))}
        bundle = export.write_text_bundle(output / f"{country}_text.jsonl.gz", projected,
                                          threshold=args.text_threshold)
        stats["text_bundle_records"] = bundle
        rows = lean_rows

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
        contract_parts[country] = export.dashboard_contract(
            {country: {**stats, "source_documents": read_count}}, {country: projected})
        # Release this country's frames before the next one is read: a full corpus otherwise sits at
        # the memory ceiling by Portugal and dies while hashing the last bundle.
        # The parent identities survive the frame release: children are attached to the *video*
        # that this export keeps, not to an ingest run the collapse dropped.
        parents = [{"record_id": row.get("record_id"), "_storage_id": row.get("_storage_id")}
                   for row in kept]
        del projected, lean_rows, raw, kept
        gc.collect()

        child_counts[country] = {}
        children_contract[country] = {}
        for role in args.children:
            name = export.child_collection(country, role)
            if name not in existing:
                # Absence is a finding: Poland has no DNA statements collection, PL/PT have no SNA.
                child_counts[country][role] = 0
                collections_read[country][role] = "absent"
                if role == "dna_statements":
                    # Coverage is a claim about the export; "the collection is absent" is itself a
                    # coverage result and must be recorded, not omitted.
                    export.reconcile_dna_coverage(contract_parts[country]["stage_coverage"][country],
                                                  [], parents, available=False)
                continue
            documents = [export.project_row(doc) for doc in
                         iter_documents(db[name], batch_size=args.batch_size, limit=args.limit)]
            attached, detail = export.attach_parent_ids(documents, parents, country=country)
            # A child table exists to be joined. Rows whose parent did not survive the collapse are
            # counted in `detail["orphan"]` and left out of the CSV rather than written unjoinable.
            documents = [row for row in attached if row.get("parent_record_id")]
            child_path = output / "children" / f"{country}_{role}.csv"
            export.write_rows(child_path, documents)
            child_outputs.append(child_path)
            # The manifest counts rows in the table, so it counts joined rows: the orphan count
            # lives in `child_detail`, where it explains the difference instead of inflating it.
            child_counts[country][role] = detail["matched"]
            # `records` must describe the table, or the auditor compares it to the file and reports drift.
            # Orphans are reported beside it, not folded into it.
            child_detail.setdefault(country, {})[role] = {
                "records": detail["matched"],
                "matched": detail["matched"],
                "orphan": detail["orphan"],
                "source_records": detail["records"],
            }
            children_contract[country][role] = export.child_contract(
                documents, country=country, role=role)
            if role == "dna_statements":
                export.reconcile_dna_coverage(contract_parts[country]["stage_coverage"][country],
                                              documents, parents, available=True)
            for document in documents:
                statement_id = str(document.get("_storage_id") or "")
                if statement_id:
                    child_statement_countries.setdefault(statement_id, set()).add(country)
            if detail["orphan"]:
                print(f"{country}: {role} {detail['records']} rows, {detail['orphan']} with no "
                      f"parent in this {args.record_unit}-unit export (excluded: a child that cannot "
                      f"join is not part of a joinable table)")
            collections_read[country][role] = name

        print(f"{country}: {read_count} read -> {stats['records']} records "
              f"({stats['duplicates_collapsed']} collapsed, {stats['quarantined']} quarantined)")

    combined.sort(key=lambda row: str(row.get("record_id", "")))
    export.write_rows(output / "combined.csv", combined)

    outputs = ([output / "combined.csv"]
               + [output / f"{c}.csv" for c in args.country]
               + [output / f"{c}_text.jsonl.gz" for c in args.country]
               # Child tables are outputs too: a file the manifest does not list is a file nobody
               # notices going missing.
               + child_outputs)
    collisions = sum(1 for countries in child_statement_countries.values() if len(countries) > 1)
    manifest = export.write_manifest(
        output,
        record_unit=args.record_unit,
        countries=country_stats,
        child_counts=child_counts,
        collections_read=collections_read,
        provenance=provenance,
        outputs=outputs,
        contract=_merge_contracts(contract_parts),
        child_detail=child_detail,
        heavy_fields=heavy_fields,
        text_threshold=args.text_threshold,
        children_contract={"unit": args.record_unit,
                           "join": "source_record_id -> parent_record_id",
                           "source_unit": "ingest", "countries": children_contract,
                           "cross_country_duplicate_statement_ids": collisions},
        heavy_splits=heavy_splits,
    )
    print(f"manifest: {manifest['records']} records, {manifest['distinct_media']} distinct media, "
          f"unit={manifest['record_unit']} -> {output}")
    for warning in manifest.get("contract_warnings") or []:
        print(f"WARNING {warning['kind']} {warning.get('country')} {warning.get('step')}: "
              f"{warning['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
