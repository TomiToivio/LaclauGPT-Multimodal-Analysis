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


def write_checkpoint(df: pd.DataFrame, csv_path: str | Path, *, stage: str, country: str) -> dict:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

    sqlite_path = path.with_suffix(".sqlite3")
    sqlite_tmp = sqlite_path.with_suffix(".sqlite3.tmp")
    if sqlite_tmp.exists():
        sqlite_tmp.unlink()
    with sqlite3.connect(sqlite_tmp) as db:
        df.to_sql("records", db, index=False, if_exists="replace")
    os.replace(sqlite_tmp, sqlite_path)

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
    os.replace(manifest_tmp, manifest_path)
    return manifest
