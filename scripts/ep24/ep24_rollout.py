"""Conservative, auditable EP24 rollout planning and non-destructive bootstrap (#373).

Reads canonical private input files; never assumes the first 500 rows are finished.
Existing Mongo records are matched by country-scoped media identity, not row offset.
No writes without --apply and an explicit sample approval marker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ep24_bootstrap import ordered_country_files, prepare_dataframe, country_slug
from ep24_stage_orchestrator import eligible_query, status_path
from ep24_settings import load_private_env
from roihu_storage import MongoStorage, StorageConfig

SAMPLE = ("finland", "poland", "portugal")
STEPS = range(1, 10)


def identity(row):
    """Source media identity; reject empty identities instead of inventing matches."""
    video = str(row.get("video_id", "") or "").strip()
    media = str(row.get("allas_filename", "") or "").strip()
    if not video or not media:
        return None
    return (video, media)


def reconcile(source_rows, existing_rows):
    """Return new rows, existing matches and QA counts; fail closed on ambiguous IDs."""
    existing = {}
    storage_ids = set()
    for doc in existing_rows:
        key = identity(doc)
        sid = str(doc.get("_storage_id", "") or "")
        if not key or not sid or key in existing or sid in storage_ids:
            raise ValueError("Existing Mongo records have missing/ambiguous media or storage IDs; manual reconciliation required")
        existing[key] = doc
        storage_ids.add(sid)
    incoming = set()
    to_insert = []
    matched = []
    duplicates = 0
    invalid = 0
    for doc in source_rows:
        key = identity(doc)
        if key is None:
            invalid += 1
            continue
        if key in incoming:
            duplicates += 1
            continue
        incoming.add(key)
        if key in existing:
            matched.append(existing[key])
        else:
            if doc["_storage_id"] in storage_ids:
                raise ValueError("Generated storage ID collides with different Mongo media identity")
            to_insert.append(doc)
    orphans = len(set(existing) - incoming)
    if orphans:
        raise ValueError(f"{orphans} existing Mongo media IDs absent from authoritative input; refusing potentially wrong namespace")
    return to_insert, matched, {"input_rows": len(source_rows), "unique_media": len(incoming),
                                "duplicate_input_rows": duplicates, "invalid_input_rows": invalid,
                                "already_in_mongo": len(matched), "new_to_insert": len(to_insert),
                                "existing_mongo_rows": len(existing_rows)}


def stage_counts(existing_rows):
    return {str(step): dict(Counter(
        str(row.get("_pipeline", {}).get(f"step_{step:02d}", {}).get("status", "pending"))
        for row in existing_rows
    )) for step in STEPS}


def discover(root, mode):
    paths = ordered_country_files(root)
    return [p for p in paths if (country_slug(p) in SAMPLE) == (mode == "continue")]


def _approval(path):
    if not path or not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise ValueError("An explicit nonempty sample approval file is required for writes/submission")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["continue", "fresh"], required=True)
    ap.add_argument("--input-root", type=Path, default=Path(os.getenv(
        "LACLAUGPT_EP24_INPUT_ROOT",
        "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess")))
    ap.add_argument("--approval-file", type=Path)
    ap.add_argument("--apply", action="store_true", help="insert missing input records only; no analysis jobs")
    ap.add_argument("--report", type=Path, help="optional private JSON audit report")
    args = ap.parse_args(argv)
    load_private_env()
    paths = discover(args.input_root, args.mode)
    if not paths:
        ap.error("No matching country inputs found")
    if args.apply:
        _approval(args.approval_file)
        if os.getenv("LACLAUGPT_MONGO_ENABLED", "1").lower() not in ("1", "true", "yes", "on"):
            ap.error("MongoDB must be enabled for rollout")
    results = []
    for path in paths:
        country = country_slug(path)
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        prepared = prepare_dataframe(source, country=country)
        records = prepared.to_dict(orient="records")
        prior_country = os.environ.get("LACLAUGPT_COUNTRY")
        os.environ["LACLAUGPT_COUNTRY"] = country
        try:
            with_storage = os.getenv("LACLAUGPT_MONGO_ENABLED", "1").lower() in ("1", "true", "yes", "on")
            if not with_storage:
                raise RuntimeError("MongoDB required to verify existing progress; refusing unverifiable plan")
            storage = MongoStorage(StorageConfig.from_env())
            try:
                coll = storage.db[storage.collection_name("dataframe")]
                existing = list(coll.find({}, {"_id": 0, "_storage_id": 1, "video_id": 1,
                     "allas_filename": 1, "_pipeline": 1}))
                new, matched, counts = reconcile(records, existing)
                if args.mode == "fresh" and existing and not args.apply:
                    counts["warning"] = "Existing records found; fresh-country run will preserve them"
                if args.mode == "fresh" and existing and args.apply:
                    raise ValueError(f"{country} already has {len(existing)} records; manual inspection required for fresh-country bootstrap")
                if args.mode == "continue" and not matched:
                    raise ValueError(f"{country}: no sample records matched authoritative source; verify dataset/collection before continuation")
                if args.apply:
                    # Upsert-by-insert ONLY: never replace model fields or _pipeline statuses.
                    from pymongo import UpdateOne
                    batch = [UpdateOne({"_storage_id": doc["_storage_id"]},
                             {"$setOnInsert": dict(doc, _pipeline={"bootstrap": {"status": "complete"}})},
                             upsert=True) for doc in new]
                    for index in range(0, len(batch), 500):
                        if batch[index:index+500]:
                            coll.bulk_write(batch[index:index+500], ordered=False)
                    if args.mode == "fresh":
                        from ep24_context import bootstrap_context
                        private_root = Path(os.getenv("LACLAUGPT_EP24_PRIVATE_ROOT",
                            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess"))
                        bootstrap_context(storage, prepared, private_root=private_root, country=country)
                    existing = list(coll.find({}, {"_id": 0, "_storage_id": 1, "video_id": 1,
                             "allas_filename": 1, "_pipeline": 1}))
                counts["stages"] = stage_counts(existing)
                counts["country"] = country
                counts["collection"] = storage.collection_name("dataframe")
                counts["action"] = "inserted_missing_only" if args.apply else "read_only_plan"
                results.append(counts)
            finally:
                storage.close()
        finally:
            if prior_country is None:
                os.environ.pop("LACLAUGPT_COUNTRY", None)
            else:
                os.environ["LACLAUGPT_COUNTRY"] = prior_country
    report = {"mode": args.mode, "countries": results, "applied": args.apply,
              "dataset": os.getenv("LACLAUGPT_DATASET", "ep2024_reprocess")}
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
        args.report.chmod(0o600)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
