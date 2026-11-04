#!/usr/bin/env python3
"""Read-only FI/PL/PT live Mongo snapshot for the Friday EP24 presentation (#373).

Run from the public repository checkout on Roihu, with private .env configured:
  python scripts/ep24/export_live_sample.py --dataset YOUR_ACTUAL_SAMPLE_DATASET

The dataset argument is deliberately mandatory: the sample and full corpus can
use DIFFERENT Mongo collection prefixes. No GPU jobs, writes or claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ep24_settings import load_private_env, private_root
from roihu_storage import MongoStorage, StorageConfig
from scripts.ep24.export_full_corpus import frame_from_docs, verified_checkpoint, coverage

COUNTRIES = ("finland", "poland", "portugal")
STEPS = range(1, 8)


def status_columns(docs):
    """Flatten stage status without replacing nested cumulative provenance."""
    result = []
    for doc in docs:
        pipeline = doc.get("_pipeline") or {}
        row = {}
        for step in STEPS:
            state = pipeline.get(f"step_{step:02d}") or {}
            row[f"step_{step:02d}_status"] = str(state.get("status") or "pending")
        result.append(row)
    return result


def presentation_frame(docs, country):
    frame = frame_from_docs(docs)
    if frame.empty:
        raise ValueError(f"{country}: no Mongo records available yet")
    if "country" not in frame:
        frame["country"] = country
    else:
        allowed = {"finland": "fi", "poland": "pl", "portugal": "pt"}
        values = frame["country"].astype(str).str.casefold()
        if not values.isin([country, allowed[country]]).all():
            raise ValueError(f"{country}: unexpected mixed country labels")
    # frame_from_docs orders by stable _storage_id, so align statuses by ID.
    by_id = {str(doc["_storage_id"]): doc for doc in docs}
    statuses = status_columns([by_id[str(key)] for key in frame["_storage_id"]])
    for key in statuses[0]:
        frame[key] = [status[key] for status in statuses]
    return frame


def snapshot_summary(docs):
    counts = coverage(docs)
    return {
        "observed_records": len(docs),
        "step_07_complete": counts["step_07"].get("complete", 0),
        "all_step_07_complete": bool(docs) and counts["step_07"].get("complete", 0) == len(docs),
        "coverage": counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="Exact LACLAUGPT_DATASET used by the active sample; never guessed")
    parser.add_argument("--country", nargs="+", choices=COUNTRIES, default=list(COUNTRIES))
    parser.add_argument("--output-root", type=Path, default=None,
                        help="PRIVATE parent directory (a new timestamped snapshot is created)")
    args = parser.parse_args(argv)
    load_private_env()
    os.environ["LACLAUGPT_DATASET"] = args.dataset
    root = (args.output_root or private_root() / "outputs" / "presentation_snapshots").expanduser().resolve()
    if root == ROOT or ROOT in root.parents:
        parser.error("Private CSVs must never be written inside the public repository")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / ("live_" + stamp + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(mode=0o700)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "snapshot_type": "INTERIM_CONCURRENT_WRITES_NOT_ATOMIC",
        "note": "Read-only sequential country snapshots; records/stages can change during export.",
        "countries": {},
    }
    frames = []
    for country in dict.fromkeys(args.country):
        old = os.environ.get("LACLAUGPT_COUNTRY")
        os.environ["LACLAUGPT_COUNTRY"] = country
        try:
            storage = MongoStorage(StorageConfig.from_env())
            try:
                collection = storage.collection_name("dataframe")
                # Queries only: never create indexes, modify statuses, claim or launch jobs.
                docs = list(storage.db[collection].find({}))
            finally:
                storage.close()
        finally:
            if old is None:
                os.environ.pop("LACLAUGPT_COUNTRY", None)
            else:
                os.environ["LACLAUGPT_COUNTRY"] = old
        frame = presentation_frame(docs, country)
        dest = destination / (country + "_live_steps_1_7.csv")
        checkpoint = verified_checkpoint(frame, dest, country=country)
        manifest["countries"][country] = {
            **snapshot_summary(docs),
            "collection": collection,
            "csv": dest.name,
            "sha256": checkpoint["csv_sha256"],
            "columns": list(frame.columns),
        }
        frames.append(frame)
        print(f"{country}: {len(frame)} persisted rows; "
              f"Step 7 complete: {manifest['countries'][country]['step_07_complete']}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.reindex(columns=sorted(combined.columns))
    combined_path = destination / "fi_pl_pt_live_combined.csv"
    check = verified_checkpoint(combined, combined_path, country="fi_pl_pt")
    manifest["combined"] = {
        "rows": len(combined), "csv": combined_path.name,
        "sha256": check["csv_sha256"], "columns": list(combined.columns),
    }
    manifest_path = destination / "live_manifest.json"
    tmp = destination / "live_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, manifest_path)
    if hashlib.sha256(combined_path.read_bytes()).hexdigest() != manifest["combined"]["sha256"]:
        raise RuntimeError("Combined CSV checksum mismatch")
    print(f"INTERIM snapshot: {destination}")
    print("These are persisted rows observed during live processing, not necessarily 500 per country.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
