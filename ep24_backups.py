"""Atomic CSV + SQLite + manifest checkpoints for cumulative EP24 dataframes."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested Mongo-derived cells to JSON for SQLite checkpointing.

    Do not mutate the source frame: MongoDB must retain structured values.
    JSON strings are readable and can be reconstructed with json.loads.
    """
    result = df.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        if not result[column].map(
            lambda value: isinstance(value, (dict, list, tuple, set))
        ).any():
            continue
        result[column] = result[column].map(
            lambda value: json.dumps(
                sorted(value, key=str) if isinstance(value, set) else value,
                ensure_ascii=False,
                default=str,
            )
            if isinstance(value, (dict, list, tuple, set))
            else value
        )
    return result


def write_checkpoint(df: pd.DataFrame, csv_path: str | Path, *, stage: str, country: str) -> dict:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe = _sqlite_safe_dataframe(df)\n    safe.to_csv(tmp, index=False, encoding="utf-8")\n    tmp.chmod(0o600)
    os.replace(tmp, path)

    sqlite_path = path.with_suffix(".sqlite3")
    sqlite_tmp = sqlite_path.with_suffix(".sqlite3.tmp")
    if sqlite_tmp.exists():
        sqlite_tmp.unlink()
    with sqlite3.connect(sqlite_tmp) as db:
        safe.to_sql("records", db, index=False, if_exists="replace")
    sqlite_tmp.chmod(0o600)\n    os.replace(sqlite_tmp, sqlite_path)

    manifest = {
        "stage": stage,
        "country": country,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "csv": str(path),
        "csv_sha256": _sha256(path),
        "sqlite": str(sqlite_path),
        "sqlite_sha256": _sha256(sqlite_path),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_tmp.chmod(0o600)\n    os.replace(manifest_tmp, manifest_path)
    return manifest
