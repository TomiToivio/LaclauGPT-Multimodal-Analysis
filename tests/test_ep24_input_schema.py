"""Tests for the EP24 incoming-schema inspector (#128 §5).

Synthetic fixtures only -- no private data is copied into this repository.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ep24_input_schema as insp  # noqa: E402
from ep24_schema import EP24_REPROCESS_COLUMNS  # noqa: E402

CANONICAL = list(EP24_REPROCESS_COLUMNS)
CLEANING_EXTRA = [
    "entities_seed_provenance", "themes_seed_provenance", "researcher_note_source",
    "cleaning_status", "cleaning_reason", "cleaning_rule_id",
    "include_in_reprocess", "needs_human_review",
]


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            # Only emit the fields this file actually declares.
            writer.writerow({c: row.get(c, "") for c in cols})


def _row(**overrides) -> dict[str, str]:
    base = {c: "" for c in [*CANONICAL, *CLEANING_EXTRA]}
    base.update({
        "video_id": "v1", "country": "Finland", "allas_filename": "a.mp4",
        "video_duration": "12.5", "author_username": "researcher",
    })
    base.update(overrides)
    return base


def _root(tmp_path: Path) -> Path:
    return tmp_path


def test_reads_a_materialized_csv_and_preserves_column_order(tmp_path):
    root = _root(tmp_path)
    cols = [*CANONICAL, *CLEANING_EXTRA]
    _write_csv(insp.country_csv_path(root, "Portugal"), cols, [_row()])
    columns, rows, digest = insp.read_rows(insp.country_csv_path(root, "Portugal"))
    assert columns == cols, "column order must be preserved exactly as stored"
    assert len(rows) == 1
    assert digest == hashlib.sha256(
        insp.country_csv_path(root, "Portugal").read_bytes()).hexdigest()


def test_refuses_git_lfs_pointer_stubs(tmp_path):
    root = _root(tmp_path)
    p = insp.country_csv_path(root, "Finland")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\nsize 12345\n", encoding="utf-8")
    with pytest.raises(insp.LfsPointerError):
        insp.read_rows(p)
    result = insp.inspect_country(root, "Finland")
    assert result.is_lfs_pointer is True
    assert result.row_count == 0


def test_missing_file_is_reported_not_raised(tmp_path):
    result = insp.inspect_country(tmp_path, "Sweden")
    assert result.exists is False
    assert result.error
    assert insp.summary([result])["missing_files"] == ["Sweden"]


def test_thirteen_column_finland_is_the_canonical_signature(tmp_path):
    root = _root(tmp_path)
    _write_csv(insp.country_csv_path(root, "Finland"), CANONICAL, [_row()])
    result = insp.inspect_country(root, "Finland")
    assert result.matches_canonical is True
    assert result.extra_columns == []
    assert result.missing_canonical_columns == []


def test_twenty_one_column_country_keeps_cleaning_fields_as_extras(tmp_path):
    """The real divergence: 8 of 10 countries carry the cleaning fields.

    They must be reported as extra incoming columns, never dropped and never
    mistaken for a broken schema.
    """
    root = _root(tmp_path)
    _write_csv(insp.country_csv_path(root, "Croatia"),
               [*CANONICAL, *CLEANING_EXTRA], [_row()])
    result = insp.inspect_country(root, "Croatia")
    assert result.matches_canonical is False
    assert result.column_count == 21
    assert result.extra_columns == CLEANING_EXTRA
    assert result.missing_canonical_columns == []
    assert not result.required_media_missing


def test_required_media_columns_are_reported_when_absent(tmp_path):
    root = _root(tmp_path)
    lacking = [c for c in CANONICAL if c != "allas_filename"]
    _write_csv(insp.country_csv_path(root, "Bulgaria"), lacking, [_row()])
    result = insp.inspect_country(root, "Bulgaria")
    assert result.required_media_missing == ["allas_filename"]
    agg = insp.summary([result])
    assert agg["usable_for_media"] == []


def test_always_empty_columns_are_detected(tmp_path):
    root = _root(tmp_path)
    _write_csv(insp.country_csv_path(root, "Spain"), CANONICAL,
               [_row(), _row(video_id="v2")])
    result = insp.inspect_country(root, "Spain")
    # researcher_new_persons/themes/note are blank in both fixture rows
    assert "researcher_note" in result.empty_columns


def test_signature_groups_countries_with_identical_headers(tmp_path):
    root = _root(tmp_path)
    _write_csv(insp.country_csv_path(root, "Finland"), CANONICAL, [_row()])
    _write_csv(insp.country_csv_path(root, "Poland"), CANONICAL, [_row()])
    _write_csv(insp.country_csv_path(root, "Croatia"),
               [*CANONICAL, *CLEANING_EXTRA], [_row()])
    rows = [insp.inspect_country(root, c) for c in ("Finland", "Poland", "Croatia")]
    agg = insp.summary(rows)
    assert agg["distinct_signatures"] == 2
    assert "Finland, Poland" in insp.render(rows)


def test_country_order_starts_with_the_three_smoke_countries():
    """#128 §6: Finland -> Poland -> Portugal must lead the iteration order."""
    assert insp.COUNTRIES[:3] == ("Finland", "Poland", "Portugal")
    assert set(insp.COUNTRIES) == set(insp.COUNTRY_TOKENS)
    assert len(insp.COUNTRIES) == 10


def test_inspection_never_writes_to_the_input(tmp_path):
    root = _root(tmp_path)
    p = insp.country_csv_path(root, "France")
    _write_csv(p, CANONICAL, [_row()])
    before = p.read_bytes()
    insp.inspect_all(root)
    assert p.read_bytes() == before, "the inspector must be read-only"


def test_no_cell_values_leak_into_the_rendered_table(tmp_path):
    """The report is shareable: it must carry names/counts/hashes, not data."""
    root = _root(tmp_path)
    secret = "RESEARCHER_NOTE_THAT_MUST_NOT_BE_PRINTED"
    _write_csv(insp.country_csv_path(root, "Hungary"), CANONICAL,
               [_row(researcher_note=secret)])
    text = insp.render(insp.inspect_all(root))
    assert secret not in text
    assert "Hungary" in text
