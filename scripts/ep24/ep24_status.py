#!/usr/bin/env python3
"""Show MongoDB EP24 progress by country and numbered stage.

A total is not a count of records (#379): duplicate rows inflate it, and a row without a stable id can
neither be resumed nor exported. So this reports the dataset and collection it actually read, the total,
the distinct ``_storage_id`` count, duplicates and rows missing an id.

This is reporting. The gate before a resubmission is ``scripts/ep24/preflight_pilot.py``, which asserts
the pilot's identity, isolation and claim state and exits non-zero; keeping a second gate here would let
the two drift apart.
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
from ep24_status_identity import identity_summary

STATES = ("claimed", "complete", "error", "retry", "skipped")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country")
    args = parser.parse_args()
    selected = [args.country.casefold()] if args.country else countries()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
