"""EP24 country bootstrap for MongoDB-first restartable processing.

Private CSV/codebook contents never belong in this public repository. This module
reads already-canonicalized private country CSVs, validates human `entities` /\n`themes`, creates stable record IDs, and stores cumulative records in MongoDB.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import pandas as pd

from ep24_backups import write_checkpoint
from ep24_context import bootstrap_context
from roihu_storage import MongoStorage, StorageConfig

LOG = logging.getLogger("ep24_bootstrap")
PRIORITY = ("finland", "poland", "portugal")


# Issue #21: private country CSVs are canonicalized before bootstrap.
# Human annotations arrive in `entities` and `themes`; bootstrap must not
# reconstruct or overwrite them.

def country_slug(path: Path) -> str:
    return path.stem.removeprefix("ep24_").casefold()


def ordered_country_files(root: Path) -> list[Path]:
    files = sorted(root.glob("ep24_*.csv"), key=lambda p: country_slug(p))
    def rank(path: Path) -> tuple[int, str]:
        slug = country_slug(path)
        try:
            return (PRIORITY.index(slug), slug)
        except ValueError:
            return (len(PRIORITY), slug)
    return sorted(files, key=rank)


def stable_ep24_id(row: pd.Series, *, country: str, row_number: int) -> str:
    parts = [
        "ep2024_reprocess",
        country,
        str(row.get("video_id", "")),
        str(row.get("source_recording", "")),
        str(row.get("sequence_number", "")),
        str(row.get("allas_filename", "")),
        str(row_number),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def prepare_dataframe(df: pd.DataFrame, *, country: str) -> pd.DataFrame:
    """Validate canonical input and add stable storage identity only."""
    out = df.copy()
    required = {"video_id", "allas_filename", "entities", "themes"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(
            f"{country}: input is not migrated to issue #21 canonical schema; missing={missing}"
        )

    forbidden = {
        "new_entity", "researcher_new_persons",
        "new_theme", "researcher_new_themes",
    }
    leaked = sorted(forbidden & set(out.columns))
    if leaked:
        raise ValueError(
            f"{country}: legacy annotation columns must be removed before Step 1: {leaked}"
        )

    out["_storage_id"] = [
        stable_ep24_id(row, country=country, row_number=i)
        for i, (_, row) in enumerate(out.iterrows())
    ]
    return out


def bootstrap_country(path: Path, *, private_root: Path, backup_root: Path, dry_run: bool = False) -> int:
    country = country_slug(path)
    LOG.info("bootstrap country=%s input=%s", country, path)
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    prepared = prepare_dataframe(source, country=country)
    write_checkpoint(prepared, backup_root / country / "step_00_bootstrap.csv",
                     stage="bootstrap", country=country)

    if dry_run:
        LOG.info("dry-run country=%s rows=%d columns=%d", country, len(prepared), len(prepared.columns))
        return len(prepared)

    old_country = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    try:
        config = StorageConfig.from_env()
        storage = MongoStorage(config)
        try:
            docs = []
            for _, row in prepared.iterrows():
                doc = row.to_dict()
                doc["_pipeline"] = {"bootstrap": {"status": "complete"}}
                docs.append(doc)
            count = storage.upsert_documents("dataframe", docs)

            context = bootstrap_context(
                storage,
                prepared,
                private_root=private_root,
                country=country,
            )
            LOG.info(
                "Mongo bootstrap country=%s upserts=%d codebooks=%d memory_seeds=%d fingerprint=%s",
                country,
                count,
                context["codebook_count"],
                context["memory_seed_count"],
                context["codebook_fingerprint"],
            )
            return count
        finally:
            storage.close()
    finally:
        if old_country is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = old_country


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
        default=Path(os.getenv("LACLAUGPT_EP24_INPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess")))
    parser.add_argument("--private-root", type=Path,
        default=Path(os.getenv(
            "LACLAUGPT_EP24_PRIVATE_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess",
        )))
    parser.add_argument("--backup-root", type=Path,
        default=Path(os.getenv("LACLAUGPT_EP24_OUTPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs")))
    parser.add_argument("--country")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    files = ordered_country_files(args.input_root)
    if args.country:
        files = [p for p in files if country_slug(p) == args.country.casefold()]
    if not files:
        raise SystemExit(f"No EP24 country CSVs found in {args.input_root}")
    total = 0
    for path in files:
        total += bootstrap_country(path, private_root=args.private_root,
                                   backup_root=args.backup_root, dry_run=args.dry_run)
    LOG.info("bootstrap complete countries=%d rows=%d", len(files), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
