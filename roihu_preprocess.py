"""EP24 Step 1: schema-preserving one-frame OCR + backend-neutral ASR.

Issue #128 contract:
* authoritative input is the materialized private per-country CSV / cumulative batch;
* preserve every incoming column in order;
* exactly one still image at original-video t=1.0s;
* exactly one OCR field, ocr_1;
* generic ASR fields, never whisper_*;
* SQLite is a job-local restart cache; MongoDB is canonical under the orchestrator;
* CSV remains a first-class compatibility artifact and gets a versioned backup.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import pandas as pd

from asr_backend import describe_backend, language_hint, load_asr_model
from ep24_pipeline import local_media_path
from ep24_schema import value as ep24_value
from ep24_video import analysis_start_seconds as video_initial_skip_seconds
from ep24_video import (
    is_too_short,
    prepare_analysis_clip,
)
from ocr_backend import describe_ocr_backend, load_ocr_backend

FRAME_TIMESTAMP_SECONDS = 1.0
# Bump when a contracted Step 1 output changes meaning (not merely when a column
# is added, which the key-derived fingerprint already covers). Existing cache
# rows become unreachable instead of being served as if still valid.
CACHE_SCHEMA_VERSION = 2
REQUIRED_MEDIA_COLUMNS = ("video_id", "allas_filename")
REQUIRED_ANNOTATION_COLUMNS = ("entities", "themes")
FORBIDDEN_LEGACY_ANNOTATION_COLUMNS = (
    "new_entity", "researcher_new_persons", "new_theme", "researcher_new_themes",
)
COUNTRY_ORDER = (
    "finland",
    "poland",
    "portugal",
    "germany",
    "spain",
    "hungary",
    "croatia",
    "france",
    "bulgaria",
    "sweden",
)
PREPROCESS_COLUMNS = (
    "frame_file",
    "frame_timestamp_seconds",
    "ocr_1",
    "ocr_backend",
    "ocr_model",
    "ocr_runtime_ms",
    "asr_transcript",
    "asr_language",
    "asr_translated",
    "asr_backend",
    "asr_model",
    "asr_runtime_ms",
    "video_duration_seconds",
    "preprocess_status",
    "preprocess_note",
    "preprocess_completed_at",
)

#: Fields the restart cache persists. Derived from the contracted Step 1 output
#: list rather than written out by hand, because the original hand-maintained
#: copy is exactly how `ocr_runtime_ms` / `asr_runtime_ms` went missing: two
#: lists that had to be kept in sync by hand drifted apart, and a cache hit then
#: produced a record with fewer fields than a fresh run. Deriving it means a
#: newly contracted output is either cached automatically or fails loudly in
#: `save_cached`, never silently omitted.
CACHE_STATUS_COLUMNS = ("preprocess_status", "preprocess_note")
CACHE_COLUMNS = tuple(
    column for column in PREPROCESS_COLUMNS if column not in CACHE_STATUS_COLUMNS
)

Path("./logs").mkdir(exist_ok=True)
Path("./database").mkdir(exist_ok=True)
Path("./Keyframes").mkdir(exist_ok=True)

LOG = logging.getLogger("roihu_preprocess")
if not LOG.handlers:
    handler = RotatingFileHandler(
        "./logs/preprocess.log", encoding="utf-8", maxBytes=5_000_000, backupCount=10
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(logging.DEBUG)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


def get_video_duration(video_filename: str) -> float:
    video = cv2.VideoCapture(video_filename)
    try:
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_filename}")
        fps = float(video.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frames <= 0:
            raise ValueError(
                f"Invalid video metadata: path={video_filename!r} fps={fps} frames={frames}"
            )
        return frames / fps
    finally:
        video.release()


def save_single_keyframe(
    video_filename: str,
    *,
    video_id: str,
    author_username: str,
    source_type: str,
) -> str:
    """Extract one and only one frame at original source t=1.0 seconds."""
    capture = cv2.VideoCapture(video_filename)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_filename}")
        capture.set(cv2.CAP_PROP_POS_MSEC, FRAME_TIMESTAMP_SECONDS * 1000.0)
        success, image = capture.read()
        if not success or image is None:
            raise ValueError(
                f"Could not read original-video frame at t={FRAME_TIMESTAMP_SECONDS:.1f}s"
            )
    finally:
        capture.release()

    platform = (source_type or "video").strip().lower()
    safe_author = (author_username or "unknown").replace("/", "_")
    safe_id = (video_id or "unknown").replace("/", "_")
    directory = Path("./Keyframes") / platform / safe_author / safe_id
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "frame_t1.0s.jpg"
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"Could not write frame: {output}")
    LOG.debug(
        "frame_saved video_id=%s timestamp=%.3f path=%s",
        video_id,
        FRAME_TIMESTAMP_SECONDS,
        output,
    )
    return str(output)


def log_schema(path: Path, df: pd.DataFrame) -> None:
    missing = [c for c in (*REQUIRED_MEDIA_COLUMNS, *REQUIRED_ANNOTATION_COLUMNS) if c not in df.columns]
    legacy = [c for c in FORBIDDEN_LEGACY_ANNOTATION_COLUMNS if c in df.columns]
    LOG.info(
        "input_schema file=%s sha256=%s rows=%d columns=%d ordered_columns=%s missing_media=%s",
        path,
        sha256_file(path),
        len(df),
        len(df.columns),
        list(df.columns),
        missing,
    )
    if legacy:
        LOG.error("legacy_annotation_columns file=%s columns=%s", path, legacy)
    LOG.debug("input_dtypes file=%s dtypes=%s", path, {c: str(t) for c, t in df.dtypes.items()})
    if missing:
        raise ValueError(
            f"Required canonical columns missing from {path}: {missing}; actual={list(df.columns)}"
        )
    if legacy:
        raise ValueError(
            f"Unmigrated legacy annotation columns in {path}: {legacy}; "
            "run the LaclauGPT-Private issue #21 migration first"
        )


def read_materialized_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not materialized data. Run git lfs pull first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    log_schema(path, df)
    return df


def connect_cache() -> sqlite3.Connection:
    # The module-level mkdir only runs for whatever CWD was current at import
    # time, so a caller that changes directory (or an orchestrator that imports
    # this stage from elsewhere) fails here with an opaque
    # "unable to open database file". Create the directory next to the database
    # we are actually about to open.
    Path("./database").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("./database/preprocess_v2.db")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS preprocess_cache (
            cache_key TEXT PRIMARY KEY,
            {", ".join(f"{column} TEXT" for column in CACHE_COLUMNS)}
        )
        """
    )
    # CREATE TABLE IF NOT EXISTS does not alter a table that already exists, so
    # a cache file written by an older checkout keeps its old shape. Add any
    # column the current schema expects, instead of failing on the next INSERT.
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(preprocess_cache)")
    }
    for column in CACHE_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE preprocess_cache ADD COLUMN {column} TEXT")
            LOG.info("cache_migrate added_column=%s", column)
    conn.commit()
    return conn


def cache_fingerprint(country: str) -> str:
    """Hash of every processing dependency that changes Step 1 output.

    A dependency change must not be able to return a stale record, so the
    fingerprint is part of the cache *key* rather than a value checked after the
    lookup: a changed backend, model, language hint or video rule simply
    addresses a different row, and the old row can never be hit again.
    """
    ocr = describe_ocr_backend()
    asr = describe_backend()
    parts = {
        "cache_schema": str(CACHE_SCHEMA_VERSION),
        "ocr_engine": str(ocr.get("engine", "")),
        "ocr_model": str(ocr.get("model", "")),
        "asr_engine": str(asr.get("engine", "")),
        "asr_model": str(asr.get("model", "")),
        "language_hint": language_hint(country),
        "frame_timestamp_seconds": f"{FRAME_TIMESTAMP_SECONDS:.3f}",
        "initial_skip_seconds": f"{video_initial_skip_seconds():.3f}",
    }
    payload = "|".join(f"{key}={value}" for key, value in sorted(parts.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(row: pd.Series, *, country: str = "") -> str:
    resolved_country = str(country or ep24_value(row, "country") or "")
    return "|".join(
        [
            resolved_country,
            str(ep24_value(row, "video_id") or ""),
            str(ep24_value(row, "allas_filename") or ""),
            cache_fingerprint(resolved_country),
        ]
    )


def load_cached(
    conn: sqlite3.Connection, key: str
) -> dict[str, str] | None:
    """Return the cached Step 1 outputs as a mapping, or None on a miss.

    The returned mapping is keyed by exactly ``CACHE_COLUMNS``, so a cache hit
    and a fresh run produce the same output field set.
    """
    columns = ", ".join(CACHE_COLUMNS)
    row = conn.execute(
        f"SELECT {columns} FROM preprocess_cache WHERE cache_key=?", (key,)
    ).fetchone()
    if row is None:
        return None
    return {column: (value or "") for column, value in zip(CACHE_COLUMNS, row)}


def save_cached(
    conn: sqlite3.Connection, key: str, values: dict[str, str]
) -> None:
    """Persist the contracted Step 1 outputs for ``key``.

    Refuses to write a partial record: if a future Stage-1 column is added to
    ``CACHE_COLUMNS`` but the caller forgets to populate it, this raises instead
    of silently storing a record that no longer matches a fresh run.
    """
    missing = [column for column in CACHE_COLUMNS if column not in values]
    if missing:
        raise KeyError(
            f"cache write is missing contracted Step 1 fields: {missing}"
        )
    columns = ", ".join(CACHE_COLUMNS)
    placeholders = ", ".join("?" for _ in CACHE_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO preprocess_cache (cache_key, {columns}) "
        f"VALUES (?, {placeholders})",
        (key, *(values[column] for column in CACHE_COLUMNS)),
    )
    conn.commit()


def backup_and_write(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output.exists():
        shutil.copy2(output, output.with_name(f"{output.name}.prewrite.{stamp}.bak"))
    df.to_csv(output, index=False, encoding="utf-8")
    backup = output.with_name(f"{output.stem}.{stamp}.backup{output.suffix}")
    shutil.copy2(output, backup)
    LOG.info(
        "output_written file=%s backup=%s sha256=%s rows=%d columns=%s",
        output,
        backup,
        sha256_file(output),
        len(df),
        list(df.columns),
    )
    return backup


def preprocess_dataframe(df: pd.DataFrame, *, source_csv: Path | None = None) -> pd.DataFrame:
    incoming_columns = list(df.columns)
    missing = [c for c in (*REQUIRED_MEDIA_COLUMNS, *REQUIRED_ANNOTATION_COLUMNS) if c not in df.columns]
    legacy = [c for c in FORBIDDEN_LEGACY_ANNOTATION_COLUMNS if c in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical input columns {missing}; actual={incoming_columns}")
    if legacy:
        raise ValueError(
            f"Legacy annotation columns must be removed before Step 1: {legacy}"
        )

    out = df.copy()
    for column in PREPROCESS_COLUMNS:
        if column not in out.columns:
            out[column] = ""

    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        out = out.head(max_rows).copy()
        LOG.info("row_limit active=%d", max_rows)

    LOG.info(
        "schema_delta input_columns=%s output_columns=%s appended=%s",
        incoming_columns,
        list(out.columns),
        [c for c in out.columns if c not in incoming_columns],
    )

    asr = load_asr_model()
    ocr = load_ocr_backend()
    LOG.info("asr_backend=%s", describe_backend())
    LOG.info("ocr_backend=%s", describe_ocr_backend())
    conn = connect_cache()
    counters = {"ok": 0, "cached": 0, "missing_video": 0, "too_short": 0, "error": 0}
    try:
        for source_row_index, (index, row) in enumerate(out.iterrows()):
            country = str(ep24_value(row, "country") or os.getenv("LACLAUGPT_COUNTRY", ""))
            video_id = str(ep24_value(row, "video_id") or "")
            author = str(ep24_value(row, "author_username") or "")
            allas = str(ep24_value(row, "allas_filename") or "")
            source_type = str(ep24_value(row, "source_type") or "video")
            local_path = str(local_media_path(row))
            key = cache_key(row)

            LOG.debug(
                "row_begin country=%s source_csv=%s source_row_index=%d video_id=%s "
                "allas_filename=%s local_video=%s",
                country,
                source_csv or "",
                source_row_index,
                video_id,
                allas,
                local_path,
            )

            cached = load_cached(conn, key)
            if cached:
                for column, value in cached.items():
                    out.at[index, column] = value
                out.at[index, "preprocess_status"] = "cached"
                out.at[index, "preprocess_note"] = (
                    "Loaded Step-1 result from SQLite restart cache "
                    f"(fingerprint={cache_fingerprint(country)})."
                )
                counters["cached"] += 1
                LOG.debug(
                    "cache_hit country=%s video_id=%s fingerprint=%s fields=%d",
                    country,
                    video_id,
                    cache_fingerprint(country),
                    len(cached),
                )
                continue

            LOG.debug("cache_miss country=%s video_id=%s", country, video_id)
            if not Path(local_path).exists():
                out.at[index, "preprocess_status"] = "missing_video"
                out.at[index, "preprocess_note"] = "Source media is not available after staging."
                counters["missing_video"] += 1
                LOG.error("missing_video country=%s video_id=%s path=%s", country, video_id, local_path)
                continue

            try:
                duration = get_video_duration(local_path)
                out.at[index, "video_duration_seconds"] = f"{duration:.6f}"
                LOG.debug("video_duration country=%s video_id=%s seconds=%.6f", country, video_id, duration)
                # Use the shared rule instead of a local `duration <` comparison.
                # ep24_video.is_too_short treats a clip of exactly 1.0s as too
                # short (no analyzable media remains after the mandatory skip),
                # while `<` classified it as work and then failed with a bogus
                # preprocess_status "error" when the frame read came back empty.
                if is_too_short(duration):
                    out.at[index, "preprocess_status"] = "too_short"
                    out.at[index, "preprocess_note"] = (
                        f"Video duration {duration:.3f}s is shorter than t={FRAME_TIMESTAMP_SECONDS:.1f}s."
                    )
                    counters["too_short"] += 1
                    continue

                frame_file = save_single_keyframe(
                    local_path,
                    video_id=video_id,
                    author_username=author,
                    source_type=source_type,
                )

                ocr_started = time.perf_counter()
                ocr_text, ocr_count = ocr.read(frame_file)
                ocr_ms = (time.perf_counter() - ocr_started) * 1000.0
                LOG.debug(
                    "ocr_complete country=%s video_id=%s backend=%s model=%s runtime_ms=%.1f "
                    "raw_result_count=%d text=%r",
                    country,
                    video_id,
                    ocr.engine,
                    ocr.model,
                    ocr_ms,
                    ocr_count,
                    ocr_text,
                )

                # ASR must not ingest the known 0-1s feed-scroll artifact.
                # Use the shared ep24_video rule and a non-destructive derived clip.
                analysis_clip = prepare_analysis_clip(local_path, "./analysis_clips")
                LOG.debug(
                    "asr_analysis_clip country=%s video_id=%s source=%s analysis_clip=%s",
                    country,
                    video_id,
                    local_path,
                    analysis_clip,
                )
                asr_started = time.perf_counter()
                result = asr.transcribe(str(analysis_clip), language_hint(country))
                asr_ms = (time.perf_counter() - asr_started) * 1000.0
                LOG.debug(
                    "asr_complete country=%s video_id=%s backend=%s model=%s runtime_ms=%.1f "
                    "language=%s transcript_chars=%d translated_chars=%d",
                    country,
                    video_id,
                    asr.engine,
                    asr.model,
                    asr_ms,
                    result.language,
                    len(result.transcript),
                    len(result.translated),
                )

                completed = datetime.now(timezone.utc).isoformat()
                values = {
                    "frame_file": frame_file,
                    "frame_timestamp_seconds": f"{FRAME_TIMESTAMP_SECONDS:.1f}",
                    "ocr_1": ocr_text,
                    "ocr_backend": ocr.engine,
                    "ocr_model": ocr.model,
                    "ocr_runtime_ms": f"{ocr_ms:.1f}",
                    "asr_transcript": result.transcript,
                    "asr_language": result.language,
                    "asr_translated": result.translated,
                    "asr_backend": asr.engine,
                    "asr_model": asr.model,
                    "asr_runtime_ms": f"{asr_ms:.1f}",
                    "video_duration_seconds": f"{duration:.6f}",
                    "preprocess_completed_at": completed,
                }
                for column, value in values.items():
                    out.at[index, column] = value
                out.at[index, "preprocess_status"] = "ok"
                out.at[index, "preprocess_note"] = (
                    "Exactly one OCR call on exactly one original-video frame at t=1.0s; "
                    "ASR processed the derived analyzable clip after the mandatory 1.0s skip."
                )
                save_cached(conn, key, values)
                counters["ok"] += 1
                LOG.debug("row_complete country=%s video_id=%s state=ok", country, video_id)
            except Exception as exc:
                counters["error"] += 1
                out.at[index, "preprocess_status"] = "error"
                out.at[index, "preprocess_note"] = f"{type(exc).__name__}: {exc}"
                LOG.exception(
                    "row_error country=%s source_row_index=%d video_id=%s path=%s",
                    country,
                    source_row_index,
                    video_id,
                    local_path,
                )
    finally:
        conn.close()

    lost = [column for column in incoming_columns if column not in out.columns]
    if lost:
        raise RuntimeError(f"Step 1 dropped incoming columns: {lost}")
    LOG.info("preprocess_counters %s", counters)
    return out


def process_csv(input_csv: Path, output_csv: Path) -> None:
    df = read_materialized_csv(input_csv)
    result = preprocess_dataframe(df, source_csv=input_csv)
    backup_and_write(result, output_csv)


def direct_country_inputs(root: Path) -> list[tuple[str, Path]]:
    pairs = []
    for country in COUNTRY_ORDER:
        path = root / f"ep24_{country}.csv"
        if path.exists():
            pairs.append((country, path))
    return pairs


def main() -> int:
    explicit = os.getenv("LACLAUGPT_INPUT_CSV")
    if explicit:
        input_csv = Path(explicit)
        output_csv = Path(os.getenv("LACLAUGPT_OUTPUT_CSV", "./csv/ep24_preprocessed.csv"))
        process_csv(input_csv, output_csv)
        return 0

    root = Path(
        os.getenv(
            "LACLAUGPT_EP24_INPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess",
        )
    )
    output_root = Path(os.getenv("LACLAUGPT_EP24_OUTPUT_ROOT", "./csv"))
    pairs = direct_country_inputs(root)
    if not pairs:
        raise FileNotFoundError(f"No materialized ep24_<country>.csv files found under {root}")

    LOG.info("country_order %s", [country for country, _ in pairs])
    for country, source in pairs:
        os.environ["LACLAUGPT_COUNTRY"] = country
        process_csv(source, output_root / country / "step_01_preprocess.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
