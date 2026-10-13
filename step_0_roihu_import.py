#!/usr/bin/env python3
"""Step 0: safely seed the EP24 Mongo dataframe from private canonical country CSVs.

Only absent documents are inserted. Existing rows (including completed steps and
researcher edits) are NEVER replaced. Repeat runs are therefore safe.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from ep24_bootstrap import ordered_country_files, prepare_dataframe, country_slug
from ep24_cli import normalize_country
from ep24_settings import load_private_env

LOG = logging.getLogger("ep24.step0")


def import_country(path: Path, *, limit: int, batch_size: int, dry_run: bool) -> tuple[int, int]:
    import pandas as pd

    country = country_slug(path)
    # Git LFS pointer files must be materialized on Roihu.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if handle.readline().startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run git -C <private-repo> lfs pull")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty country CSV: {path}")
    old_country = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    inserted = existing = 0
    storage = None
    try:
        from roihu_storage import MongoStorage, StorageConfig

        cfg = StorageConfig.from_env()
        collection_name = cfg.collection("dataframe")
        if not dry_run:
            if not cfg.mongo_enabled:
                raise RuntimeError("Step 0 needs LACLAUGPT_MONGO_ENABLED=1")
            storage = MongoStorage(cfg)
        LOG.info("step=0 country=%s source=%s database=%s collection=%s dry_run=%s",
                 country, path, cfg.mongo_database, collection_name, dry_run)
        for chunk in pd.read_csv(path, dtype=str, keep_default_na=False,
                                 chunksize=batch_size, encoding="utf-8-sig"):
            # Stable IDs in the existing bootstrap contract include the source row index.
            # Preserve its absolute offset across CSV chunks.
            start = inserted + existing
            if limit > 0:
                chunk = chunk.head(limit - start)
            if chunk.empty:
                break
            # prepare_dataframe enumerates from zero, so set the absolute index
            # before hashing, without changing input columns or researcher annotations.
            prepared = prepare_dataframe_with_offset(chunk, country=country, offset=start)
            if dry_run:
                existing += len(prepared)
                LOG.info("step=0 dry_run country=%s validated=%d", country, existing)
            else:
                from pymongo import UpdateOne

                operations = []
                now = datetime.now(timezone.utc).isoformat()
                for _, row in prepared.iterrows():
                    document = row.to_dict()
                    document["_pipeline"] = {"bootstrap": {"status": "complete", "imported_at": now}}
                    operations.append(UpdateOne(
                        {"_storage_id": document["_storage_id"]},
                        {"$setOnInsert": document},
                        upsert=True,
                    ))
                result = storage.db[collection_name].bulk_write(operations, ordered=False)
                inserted += result.upserted_count
                existing += len(operations) - result.upserted_count
                LOG.info("step=0 country=%s batch=%d inserted=%d existing=%d",
                         country, len(operations), inserted, existing)
            if limit > 0 and inserted + existing >= limit:
                break
        LOG.info("step=0 country=%s complete inserted=%d existing_or_validated=%d",
                 country, inserted, existing)
        return inserted, existing
    finally:
        if storage is not None:
            storage.close()
        if old_country is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = old_country


def prepare_dataframe_with_offset(chunk, *, country: str, offset: int):
    """Use the exact legacy bootstrap ID formula with the absolute source row."""
    from ep24_bootstrap import stable_ep24_id

    prepared = prepare_dataframe(chunk, country=country)
    prepared["_storage_id"] = [
        stable_ep24_id(row, country=country, row_number=offset + i)
        for i, (_, row) in enumerate(prepared.iterrows())
    ]
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", "-c", help="Finland, Poland, Portugal, or another private country")
    parser.add_argument("--limit", "-n", type=int, default=0, help="0 means all rows")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be >= 0 and --batch-size must be > 0")
    load_private_env()
    root = args.input_root or Path(os.environ["LACLAUGPT_EP24_INPUT_ROOT"])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    files = ordered_country_files(root)
    if args.country:
        wanted = normalize_country(args.country)
        files = [p for p in files if country_slug(p) == wanted]
    if not files:
        parser.error(f"No matching ep24_<country>.csv files in {root}")
    inserted = existing = 0
    for path in files:
        a, b = import_country(path, limit=args.limit, batch_size=args.batch_size, dry_run=args.dry_run)
        inserted += a
        existing += b
    LOG.info("step=0 FINISHED countries=%d inserted=%d existing_or_validated=%d dry_run=%s",
             len(files), inserted, existing, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
