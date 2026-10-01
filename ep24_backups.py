"""EP24 backups: atomic CSV snapshots, SQLite checkpoints, checksum manifests (#64).

The issue is explicit about the recovery contract:

    never destroy an older good checkpoint before the new one is safely written

so every snapshot here is written to a temporary path, fsynced, and then
atomically renamed into place. A crash mid-write leaves the previous good file
untouched.

CSV is a first-class research artifact, not a debug dump: the cumulative
dataframe is written with every column it carries, and a sidecar manifest records
the row count, column list, checksums and provenance so a reader can tell what
they are looking at. SQLite is a *checkpoint* of the stage state, explicitly not a
second source of truth.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SCHEMA_VERSION = "ep24-backup/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file in the same directory, then rename.

    Same-directory temp is required: `os.replace` is only atomic within a
    filesystem, and the destination must never be a partially written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass
class BackupManifest:
    """What a snapshot contains, so a reader can trust it without the pipeline."""

    country: str
    step: int
    csv_path: str
    rows: int
    columns: list[str]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    csv_sha256: str = ""
    sqlite_path: str = ""
    sqlite_sha256: str = ""
    run_provenance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"


def write_cumulative_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    country: str,
    step: int,
    run_provenance: dict[str, Any] | None = None,
    notes: str = "",
    write_manifest: bool = True,
) -> BackupManifest:
    """Atomically snapshot the cumulative dataframe and write its manifest.

    The dataframe is written with every column it has, values as strings, with no
    NaN coercion -- ``keep_default_na=False`` and ``na_rep=""`` so a blank
    researcher field stays blank rather than becoming the literal ``"nan"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp_name, index=False, na_rep="", encoding="utf-8")
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise

    manifest = BackupManifest(
        country=country,
        step=step,
        csv_path=str(path),
        rows=len(df),
        columns=[str(c) for c in df.columns],
        csv_sha256=sha256_file(path),
        run_provenance=dict(run_provenance or {}),
        notes=notes,
    )
    if write_manifest:
        _atomic_write_text(path.with_suffix(path.suffix + ".manifest.json"), manifest.to_json())
    return manifest


def write_sqlite_checkpoint(
    path: str | Path,
    *,
    stage_state: Iterable[dict[str, Any]],
    manifest: BackupManifest | None = None,
) -> str:
    """Atomically write a SQLite checkpoint of stage state. Returns its sha256.

    Explicitly a checkpoint, not a competing source of truth: MongoDB remains
    canonical, and this exists so a run can be inspected or resumed when Mongo is
    unreachable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        connection = sqlite3.connect(tmp_name)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_state (
                    record_id     TEXT NOT NULL,
                    stage         INTEGER NOT NULL,
                    state         TEXT NOT NULL,
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    error         TEXT NOT NULL DEFAULT '',
                    completed_at  REAL,
                    provenance    TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (record_id, stage)
                )
                """
            )
            for record in stage_state:
                connection.execute(
                    "INSERT OR REPLACE INTO stage_state "
                    "(record_id, stage, state, attempts, error, completed_at, provenance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.get("record_id", ""),
                        int(record.get("stage", 0)),
                        record.get("state", "pending"),
                        int(record.get("attempts", 0)),
                        record.get("error", ""),
                        record.get("completed_at"),
                        json.dumps(record.get("provenance", {}), sort_keys=True),
                    ),
                )
            connection.commit()
            # Integrity check before the file is promoted: a checkpoint that does
            # not pass its own check must not replace a good one.
            check = connection.execute("PRAGMA integrity_check").fetchone()
            if not check or check[0] != "ok":
                raise RuntimeError(f"sqlite integrity check failed: {check}")
        finally:
            connection.close()
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise

    digest = sha256_file(path)
    if manifest is not None:
        manifest.sqlite_path = str(path)
        manifest.sqlite_sha256 = digest
    return digest


def read_manifest(csv_path: str | Path) -> dict[str, Any]:
    """Load the sidecar manifest for a snapshot, or return {} when absent."""
    manifest_path = Path(str(csv_path) + ".manifest.json")
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_snapshot(csv_path: str | Path) -> tuple[bool, str]:
    """Check a snapshot against its manifest. Returns (ok, reason).

    This is what makes a backup usable during an incident: it answers "is this
    file the one the manifest describes" without re-running anything.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return False, f"missing snapshot: {csv_path}"
    manifest = read_manifest(csv_path)
    if not manifest:
        return False, f"missing manifest for {csv_path}"
    actual = sha256_file(csv_path)
    expected = manifest.get("csv_sha256", "")
    if actual != expected:
        return False, f"checksum mismatch for {csv_path.name}"
    return True, f"ok ({manifest.get('rows')} rows, {len(manifest.get('columns', []))} columns)"
