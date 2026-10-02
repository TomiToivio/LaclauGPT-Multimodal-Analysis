"""Regression tests for the Step 1 restart cache (issue #168).

The defect these lock down: the cache table, its `SELECT` and its `INSERT` were
three hand-written column lists. `ocr_runtime_ms` and `asr_runtime_ms` were
absent from all three while a fresh run wrote them, so a *cache hit* produced a
Step 1 record with two fewer fields than a *fresh run* -- the same source row,
two different outputs, depending only on whether it had been processed before.

The fix derives the cached column set from the contracted Step 1 output list, so
the two can no longer be maintained separately. These tests assert that
derivation and the end-to-end equivalence it buys, rather than the mechanics of
the current implementation, so a future rewrite of the cache is free as long as
the invariant survives.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import roihu_preprocess as rp  # noqa: E402

#: A fully-populated fresh-run cache payload. Every contracted cached field must
#: be present; `save_cached` refuses a partial write, which is itself a test.
FRESH_VALUES: dict[str, str] = {
    "frame_file": "./Keyframes/tiktok/author/vid/frame_t1.0s.jpg",
    "frame_timestamp_seconds": "1.0",
    "ocr_1": "SOME OCR TEXT",
    "ocr_backend": "paddleocr",
    "ocr_model": "PP-OCRv5",
    "ocr_runtime_ms": "123.4",
    "asr_transcript": "a transcript",
    "asr_language": "fi",
    "asr_translated": "a translation",
    "asr_backend": "canary",
    "asr_model": "nvidia/canary-1b",
    "asr_runtime_ms": "456.7",
    "video_duration_seconds": "12.000000",
    "preprocess_completed_at": "2026-10-02T00:00:00Z",
}


@pytest.fixture()
def cache_db(tmp_path: Path) -> sqlite3.Connection:
    """A cache table with the current schema, independent of the real cache file."""
    conn = sqlite3.connect(str(tmp_path / "cache.db"))
    columns = ", ".join(f"{column} TEXT" for column in rp.CACHE_COLUMNS)
    conn.execute(
        f"CREATE TABLE preprocess_cache (cache_key TEXT PRIMARY KEY, {columns})"
    )
    conn.commit()
    yield conn
    conn.close()


def test_cache_covers_every_contracted_step_1_field_except_status():
    """The cache must persist every Step 1 output, so a hit cannot be thinner.

    `preprocess_status` and `preprocess_note` describe *how* the row was
    produced ("cached" vs "ok"), so they are set by the stage rather than cached.
    """
    expected = [
        column
        for column in rp.PREPROCESS_COLUMNS
        if column not in ("preprocess_status", "preprocess_note")
    ]
    assert list(rp.CACHE_COLUMNS) == expected


def test_runtime_fields_are_cached():
    """The exact fields the original defect dropped."""
    assert "ocr_runtime_ms" in rp.CACHE_COLUMNS
    assert "asr_runtime_ms" in rp.CACHE_COLUMNS


def test_cache_hit_returns_the_same_field_set_as_a_fresh_run(cache_db):
    """Fresh and cached outputs are schema-equivalent."""
    rp.save_cached(cache_db, "fi|v1|a.mp4|fp", FRESH_VALUES)
    loaded = rp.load_cached(cache_db, "fi|v1|a.mp4|fp")

    assert loaded is not None
    assert sorted(loaded) == sorted(FRESH_VALUES)
    assert [c for c in rp.CACHE_COLUMNS if c not in loaded] == []


@pytest.mark.parametrize("field", ["ocr_runtime_ms", "asr_runtime_ms"])
def test_runtime_field_survives_the_cache_round_trip(cache_db, field):
    """Named explicitly: this is the reported bug, so keep it visible."""
    rp.save_cached(cache_db, "fi|v1|a.mp4|fp", FRESH_VALUES)
    loaded = rp.load_cached(cache_db, "fi|v1|a.mp4|fp")
    assert loaded[field] == FRESH_VALUES[field]


def test_cache_miss_returns_none(cache_db):
    assert rp.load_cached(cache_db, "fi|nope|none.mp4|fp") is None


def test_a_partial_write_is_refused(cache_db):
    """A future contracted field must fail loudly, not vanish silently.

    This is the guard that makes the original bug non-recurring: if someone adds
    a Stage 1 output to the contract and forgets the cache payload, the next
    write raises instead of storing a record that no longer matches a fresh run.
    """
    incomplete = {"frame_file": "x.jpg"}
    with pytest.raises(KeyError) as excinfo:
        rp.save_cached(cache_db, "fi|v1|a.mp4|fp", incomplete)
    assert "ocr_runtime_ms" in str(excinfo.value)

    # and nothing was written
    assert rp.load_cached(cache_db, "fi|v1|a.mp4|fp") is None


def test_fingerprint_is_stable_for_the_same_configuration():
    assert rp.cache_fingerprint("finland") == rp.cache_fingerprint("finland")


def test_fingerprint_separates_countries():
    """Country drives the language hint, so it is a processing dependency."""
    assert rp.cache_fingerprint("finland") != rp.cache_fingerprint("poland")


def test_fingerprint_changes_when_a_backend_model_changes(monkeypatch):
    """A changed model must not be able to serve a stale cached record."""
    before = rp.cache_fingerprint("finland")
    monkeypatch.setenv("LACLAUGPT_OCR_MODEL", "some-other-model")
    assert rp.cache_fingerprint("finland") != before


def test_fingerprint_changes_when_the_asr_engine_changes(monkeypatch):
    before = rp.cache_fingerprint("finland")
    monkeypatch.setenv("LACLAUGPT_ASR_ENGINE", "whisper")
    assert rp.cache_fingerprint("finland") != before


def test_fingerprint_changes_when_the_video_skip_rule_changes(monkeypatch):
    """Changing the mandatory initial skip changes what ASR receives.

    The skip rule is read into a module constant at import time, and
    ``roihu_preprocess`` imports the accessor *from* ``ep24_video``. Reloading
    only the stage is not enough: the already-imported ``ep24_video`` keeps the
    old value, so the dependency has to be reloaded first (verified directly --
    stage-only reload reports no change, both-in-order reports a change).
    """
    import importlib

    import ep24_video

    before = rp.cache_fingerprint("finland")
    monkeypatch.setenv("LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS", "2.0")
    importlib.reload(ep24_video)
    importlib.reload(rp)
    try:
        assert rp.cache_fingerprint("finland") != before
    finally:
        monkeypatch.delenv("LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS", raising=False)
        importlib.reload(ep24_video)
        importlib.reload(rp)


def test_cache_key_includes_identity_and_fingerprint():
    row = pd.Series(
        {"country": "finland", "video_id": "v1", "allas_filename": "a.mp4"}
    )
    key = rp.cache_key(row)
    assert key.split("|")[:3] == ["finland", "v1", "a.mp4"]
    assert key.split("|")[3] == ""
    assert key.split("|")[4] == rp.cache_fingerprint("finland")


def test_cache_key_separates_records_within_a_country():
    a = rp.cache_key(pd.Series({"country": "fi", "video_id": "1", "allas_filename": "a"}))
    b = rp.cache_key(pd.Series({"country": "fi", "video_id": "2", "allas_filename": "b"}))
    assert a != b


def test_existing_pre_fix_cache_table_is_migrated(tmp_path, monkeypatch):
    """A cache file written before the fix must not break the stage.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so without an
    explicit migration the next INSERT would fail on the missing columns. The
    cache is a disposable restart artifact, but silently failing every write is
    worse than upgrading it in place.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "database").mkdir(exist_ok=True)
    legacy = sqlite3.connect(str(tmp_path / "database" / "preprocess_v2.db"))
    legacy.execute(
        "CREATE TABLE preprocess_cache (cache_key TEXT PRIMARY KEY, frame_file TEXT, "
        "frame_timestamp_seconds TEXT, ocr_1 TEXT, ocr_backend TEXT, ocr_model TEXT, "
        "asr_transcript TEXT, asr_language TEXT, asr_translated TEXT, asr_backend TEXT, "
        "asr_model TEXT, video_duration_seconds TEXT, completed_at TEXT)"
    )
    legacy.commit()
    legacy.close()

    conn = rp.connect_cache()
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(preprocess_cache)")}
        assert "ocr_runtime_ms" in columns
        assert "asr_runtime_ms" in columns
        # and the upgraded table accepts a full write
        rp.save_cached(conn, "fi|v1|a.mp4|fp", FRESH_VALUES)
        assert rp.load_cached(conn, "fi|v1|a.mp4|fp")["asr_runtime_ms"] == "456.7"
    finally:
        conn.close()


def test_cache_key_changes_when_source_media_hash_changes():
    row = pd.Series(
        {"country": "finland", "video_id": "v1", "allas_filename": "a.mp4"}
    )
    first = rp.cache_key(row, media_sha256="aaa")
    second = rp.cache_key(row, media_sha256="bbb")
    assert first != second
    assert "|aaa|" in first
    assert "|bbb|" in second
