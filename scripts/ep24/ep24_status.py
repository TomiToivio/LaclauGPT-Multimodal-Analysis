#!/usr/bin/env python3
"""Show MongoDB EP24 progress by country and numbered stage."""
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
            total = coll.count_documents({})
            print(f"{country}: total={total}")
            for step in range(1, 10):
                counts = {
                    state: coll.count_documents({f"{status_path(step)}.status": state})
                    for state in ("claimed", "complete", "error", "retry", "skipped")
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
