#!/usr/bin/env python3
"""Full, lossless EP24 dataframe export with stage coverage and round-trip checks (#373).

Unlike the researcher dashboard's lean CSV / text-bundle split, this export keeps
every column and every source record, including partial/error stage statuses.
All files are private. It never writes to MongoDB or the source CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ep24_backups import write_checkpoint
from ep24_bootstrap import ordered_country_files, country_slug
from ep24_settings import load_private_env
from ep24_pipeline import json_safe_cell
from roihu_storage import MongoStorage, StorageConfig

STAGES = range(1, 8)


def frame_from_docs(documents):
    """Keep all document values (including original source fields and nested provenance)."""
    rows = [{k: v for k, v in doc.items() if k != "_id"} for doc in documents]
    if not rows:
        return pd.DataFrame()
    if any(not str(row.get("_storage_id", "") or "").strip() for row in rows):
        raise ValueError("Cannot export documents with missing _storage_id")
    rows.sort(key=lambda row: str(row["_storage_id"]))
    ids = [str(row["_storage_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate _storage_id in canonical Mongo collection")
    df = pd.DataFrame(rows)
    front = [c for c in ("_storage_id", "country", "video_id", "allas_filename") if c in df]
    return df[front + sorted(c for c in df if c not in front)]


def coverage(documents):
    result = {}
    for step in STAGES:
        field = f"step_{step:02d}"
        counts = {}
        for doc in documents:
            status = doc.get("_pipeline", {}).get(field, {}).get("status", "pending")
            counts[str(status)] = counts.get(str(status), 0) + 1
        result[field] = counts
    return result


def completed(documents):
    return all(
        str(doc.get("_pipeline", {}).get("step_07", {}).get("status", "")) == "complete"
        for doc in documents
    ) and bool(documents)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_checkpoint(frame, destination, *, country):
    manifest = write_checkpoint(frame, destination, stage="step_07_full_corpus", country=country)
    # Validate CSV contents as well as its hash: round-trip identity and columns.
    actual = pd.read_csv(destination, dtype=str, keep_default_na=False)
    if len(actual) != len(frame) or set(actual.columns) != set(frame.columns):
        raise RuntimeError(f"{country}: CSV round-trip row/column mismatch")
    expected = frame["_storage_id"].astype(str).tolist()
    if actual["_storage_id"].tolist() != expected:
        raise RuntimeError(f"{country}: CSV round-trip identity mismatch")
    for key in ("_pipeline", "_provenance"):
        if key in actual:
            for original, encoded in zip(frame[key], actual[key]):
                if json.loads(encoded) != json.loads(json_safe_cell(original)):
                    raise RuntimeError(f"{country}: nested field round-trip failure: {key}")
    if checksum(destination) != manifest["csv_sha256"]:
        raise RuntimeError("CSV checksum mismatch")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=Path(os.getenv(
        "LACLAUGPT_EP24_INPUT_ROOT",
        "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess")))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="create auditable interim snapshots, not final-complete results")
    ap.add_argument("--country", nargs="*", help="optional subset (otherwise all private input countries)")
    args = ap.parse_args(argv)
    load_private_env()
    # Never put identifiable row exports under this public checkout.
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        ap.error("Refusing to write private corpus exports into the public repository")
    selected = [country_slug(p) for p in ordered_country_files(args.input_root)]
    if args.country:
        wanted = set(args.country)
        unknown = wanted - set(selected)
        if unknown:
            ap.error(f"Unknown countries: {sorted(unknown)}")
        selected = [c for c in selected if c in wanted]
    if not selected:
        ap.error("No source input countries found")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    combined = []
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": os.getenv("LACLAUGPT_DATASET", "ep2024_reprocess"),
                "countries": {}, "all_complete": True}
    for country in selected:
        old = os.environ.get("LACLAUGPT_COUNTRY")
        os.environ["LACLAUGPT_COUNTRY"] = country
        try:
            storage = MongoStorage(StorageConfig.from_env())
            try:
                docs = list(storage.db[storage.collection_name("dataframe")].find({}))
                if not docs:
                    raise ValueError(f"{country}: no records in Mongo dataframe collection")
                is_complete = completed(docs)
                if not is_complete and not args.allow_incomplete:
                    raise ValueError(f"{country}: Step 7 incomplete; use --allow-incomplete for interim export")
                frame = frame_from_docs(docs)
                if "country" not in frame:
                    frame["country"] = country
                else:
                    values = frame["country"].astype(str).str.casefold()
                    if not values.isin((country, country[:2])).all():
                        raise ValueError(f"{country}: mixed/incorrect country values")
                country_path = output / f"{country}_full_steps_1_7.csv"
                detail = verified_checkpoint(frame, country_path, country=country)
                detail.update({"complete": is_complete, "coverage": coverage(docs),
                               "source_collection": storage.collection_name("dataframe")})
                manifest["countries"][country] = detail
                combined.append(frame)
                manifest["all_complete"] &= is_complete
            finally:
                storage.close()
        finally:
            if old is None:
                os.environ.pop("LACLAUGPT_COUNTRY", None)
            else:
                os.environ["LACLAUGPT_COUNTRY"] = old
    union = pd.concat(combined, ignore_index=True, sort=False)
    union = union.reindex(columns=sorted(union.columns))
    full = verified_checkpoint(union, output / "all_countries_full_steps_1_7.csv", country="all")
    manifest["combined"] = full
    manifest_path = output / "full_export_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(manifest_path)
    print(json.dumps({"countries": selected, "complete": manifest["all_complete"],
                      "rows": len(union), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
