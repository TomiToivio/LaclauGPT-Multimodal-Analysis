#!/usr/bin/env python3
"""Read-only EP24 country/stage status inventory for issue #306.

This is a *status* audit, not evidence of valid analysis output. Run only against
authorized MongoDB, write reports to authorized private storage, and publish
only aggregate results after disclosure review.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("complete", "claimed", "error", "retry", "skipped", "pending")
COLLECTION_RE = re.compile(
    r"^laclaugpt_ep2024_reprocess_([a-z][a-z0-9]*)_dataframe$", re.I
)


def dataframe_collections(names):
    """Only live primary dataframe collections, never reset/archive tables."""
    return {match.group(1).lower(): name for name in sorted(names)
            if (match := COLLECTION_RE.fullmatch(name))}


def tally(rows):
    """Status tallies with explicit missingness and eligibility indicators."""
    counters = [Counter() for _ in range(10)]
    total = 0
    valid_id = 0
    with_complete_chain = 0
    run_ids = set()
    for row in rows:
        total += 1
        identity = any(isinstance(row.get(key), str) and row[key].strip()
                       for key in ("video_id", "allas_filename", "_storage_id"))
        if identity:
            valid_id += 1
        counters[0]["imported"] += 1
        counters[0]["identity_valid" if identity else "identity_missing"] += 1
        provenance = row.get("_provenance")
        if isinstance(provenance, dict) and provenance.get("run_id"):
            run_ids.add(str(provenance["run_id"]))
        pipeline = row.get("_pipeline") or {}
        if not isinstance(pipeline, dict):
            pipeline = {}
        chain = identity
        for stage in range(1, 10):
            data = pipeline.get(f"step_{stage:02d}") or {}
            state = data.get("status") if isinstance(data, dict) else None
            state = str(state).lower() if state else "missing"
            counters[stage][state] += 1
            if state == "complete" and chain:
                counters[stage]["upstream_chain_complete"] += 1
            else:
                chain = False
        if chain:
            with_complete_chain += 1
    return {
        "records_read": total,
        "identity_valid": valid_id,
        "identity_missing": total - valid_id,
        "all_steps_status_complete_with_valid_identity": with_complete_chain,
        "run_ids": sorted(run_ids),
        "stage_counts": [{k: v for k, v in sorted(counter.items())}
                         for counter in counters],
        "output_payloads_verified": False,
        "run_coherence_verified": False,
    }


def scan(db, name, batch_size):
    coll = db[name]
    # Independent counts detect metadata/empty-read anomalies. Not definitive
    # without checking readable records; failures are recorded as unknown.
    meta = {"collection": name}
    for field, fn in (("estimated_documents", coll.estimated_document_count),
                      ("count_documents", lambda: coll.count_documents({}))):
        try:
            meta[field] = fn()
        except Exception:
            meta[field] = None
            meta[field + "_error"] = "query_failed"
    projection = {"_id": 0, "video_id": 1, "allas_filename": 1,
                  "_storage_id": 1, "_provenance.run_id": 1, "_pipeline": 1}
    try:
        results = tally(coll.find({}, projection).batch_size(batch_size))
        meta.update(results)
        meta["read_status"] = "ok"
        if meta["records_read"] == 0 and any(
            (meta.get(field) or 0) > 0
            for field in ("estimated_documents", "count_documents")
        ):
            meta["read_status"] = "metadata_nonzero_but_cursor_empty"
    except Exception:
        meta["read_status"] = "read_failed"
        meta["read_error"] = "query_failed"
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="authorized private directory")
    parser.add_argument("--countries", nargs="*", default=[],
                        help="optional country slugs; default: all discovered")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--timeout-ms", type=int, default=12000)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    uri = os.getenv("LACLAUGPT_MONGO_URI") or os.getenv("LACLAUGPT_MONGODB_URI")
    if not uri:
        parser.error("Mongo URI not configured in private environment")
    try:
        from pymongo import MongoClient
    except ImportError:
        parser.error("pymongo is required")
    from urllib.parse import urlsplit, unquote
    dbname = (os.getenv("LACLAUGPT_MONGO_DATABASE")
              or os.getenv("LACLAUGPT_MONGODB_DATABASE")
              or unquote(urlsplit(uri).path.lstrip("/")))
    if not dbname:
        parser.error("Mongo database must be configured in private environment")
    client = MongoClient(uri, serverSelectionTimeoutMS=args.timeout_ms,
                         retryReads=True)
    try:
        db = client[dbname]
        all_names = db.list_collection_names()
        selected = dataframe_collections(all_names)
        if args.countries:
            requested = set(c.lower() for c in args.countries)
            selected = {k: v for k, v in selected.items() if k in requested}
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "read_only_status_inventory",
            "database_name_withheld": True,
            "countries": {country: scan(db, collection, args.batch_size)
                          for country, collection in sorted(selected.items())},
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "country_stage_status.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (args.output / "country_stage_status.csv").open(
            "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=[
                "country", "stage", "records_read", "state", "count", "read_status"])
            writer.writeheader()
            for country, summary in report["countries"].items():
                for stage, counts in enumerate(summary.get("stage_counts", [])):
                    for state, count in sorted(counts.items()):
                        writer.writerow({"country": country, "stage": stage,
                                         "records_read": summary["records_read"],
                                         "state": state, "count": count,
                                         "read_status": summary["read_status"]})
        print("Country stage status audit written to private output directory.")
        return 0
    except Exception:
        print("Country stage audit failed; check private runtime diagnostics.",
              file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
