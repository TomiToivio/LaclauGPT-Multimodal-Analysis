#!/usr/bin/env python3
"""Show MongoDB EP24 progress by country and numbered stage.

The issue-#379 runbook has a hard stop: "verify each country reports the expected 500 unique pilot
records (the simple status command shows total, not uniqueness)". A total is not a count of records —
duplicate rows inflate it, and rows without a stable id cannot be resumed or exported. So this reports
the dataset and collection it actually read, the total, the distinct ``_storage_id`` count, duplicates
and rows missing an id, and it can refuse to pass silently (``--expect``).
"""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ep24_settings import load_private_env
load_private_env()

from ep24_stage_orchestrator import countries, status_path
from roihu_storage import MongoStorage, StorageConfig

STATES = ("claimed", "complete", "error", "retry", "skipped")


def identity_summary(collection) -> dict:
    """Total vs distinct stable ids, without assuming the collection is clean.

    ``collection`` is anything that answers ``count_documents`` and ``distinct`` — a real pymongo
    collection or a test double.
    """
    total = collection.count_documents({})
    ids = [value for value in collection.distinct("_storage_id") if value not in (None, "")]
    missing = collection.count_documents({"_storage_id": {"$in": [None, ""]}})
    unique = len(ids)
    return {
        "total": total,
        "unique": unique,
        "duplicates": max(total - unique - missing, 0),
        "missing_id": missing,
    }


def identity_failures(country: str, summary: dict, expect: int | None = None) -> list[str]:
    """Why this country must not pass the runbook's hard stop. Empty list means it passed."""
    failures: list[str] = []
    if expect is not None and summary["unique"] != expect:
        failures.append(
            f"{country}: expected {expect} distinct ids, found {summary['unique']} "
            f"(total={summary['total']}, duplicates={summary['duplicates']}, "
            f"missing_id={summary['missing_id']})"
        )
    if summary["duplicates"] or summary["missing_id"]:
        failures.append(
            f"{country}: {summary['duplicates']} duplicate rows and "
            f"{summary['missing_id']} rows without a stable id"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country")
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="expected number of distinct stable ids per country; exit 2 when it does not match",
    )
    args = parser.parse_args()
    selected = [args.country.casefold()] if args.country else countries()
    failed: list[str] = []
    for country in selected:
        old = os.environ.get("LACLAUGPT_COUNTRY")
        os.environ["LACLAUGPT_COUNTRY"] = country
        storage = MongoStorage(StorageConfig.from_env())
        try:
            coll = storage.db[storage.collection_name("dataframe")]
            print(f"{country}: dataset={storage.config.dataset} collection={coll.name}")
            summary = identity_summary(coll)
            print(
                f"{country}: total={summary['total']} unique={summary['unique']} "
                f"duplicates={summary['duplicates']} missing_id={summary['missing_id']}"
            )
            failed.extend(identity_failures(country, summary, args.expect))
            for step in range(1, 10):
                counts = {
                    state: coll.count_documents({f"{status_path(step)}.status": state})
                    for state in STATES
                }
                print(f"  step {step}: " + " ".join(f"{k}={v}" for k, v in counts.items()))
        finally:
            storage.close()
            if old is None:
                os.environ.pop("LACLAUGPT_COUNTRY", None)
            else:
                os.environ["LACLAUGPT_COUNTRY"] = old
    if failed:
        print("\nHARD STOP:", file=sys.stderr)
        for line in failed:
            print(f"  - {line}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
