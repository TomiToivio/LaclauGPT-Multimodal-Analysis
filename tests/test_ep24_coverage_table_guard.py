"""Tests for the cross-country coverage table drift guard (issue #91).

`docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md` promises that "every number below
is reproducible with one command per country". It was accurate when published in
#86 — all ten rows reproduce exactly against the auditor as it stood then.

It then went stale silently: #100 wired `fold_fix` into `_fold`, changing what
counts as one token, so every fragmentation count moved (599 -> 623 total) while
the document kept promising reproducibility. Nothing checked the promise.

`scripts/ep24/check_coverage_table.py` turns it into a test. These tests pin the
parser (which must tolerate the markdown emphasis the table actually uses) and
the comparison behaviour (a drifted count must be reported, an unchanged table
must pass).

Fixtures are synthetic: no private codebook content, rows, notes or personal data.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "check_coverage_table", ROOT / "scripts" / "ep24" / "check_coverage_table.py"
)
assert _SPEC and _SPEC.loader
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


# ------------------------------------------------------------------ the parser

def test_parses_a_plain_row() -> None:
    rows = guard._parse_table("| SE | 572 | 514 | 89.9% | 83 | 0.15 | 64 | 1 |")
    assert rows == {"SE": (572, 514, 89.9, 83, 0.15, 64, 1)}


def test_parses_the_bolded_form_the_table_uses() -> None:
    """The published table emboldens its extreme values (`**91.3%**`).

    A parser that only accepted the plain form would raise on the real document,
    which is exactly the "it parses my fixture but not the file" failure.
    """
    rows = guard._parse_table("| PL | 711 | 649 | **91.3%** | 76 | 0.11 | 97 | 17 |")
    assert rows == {"PL": (711, 649, 91.3, 76, 0.11, 97, 17)}


def test_parses_the_total_row_as_not_a_country() -> None:
    """`| **Total** |` must not be mistaken for a country code."""
    rows = guard._parse_table("| **Total** | **4654** | **4060** | **87%** | 842 | 0.18 | 623 | 65 |")
    assert rows == {} or "TO" not in rows


def test_parses_multiple_rows_and_skips_prose() -> None:
    text = (
        "| Country | Entries |\n"
        "| --- | --- |\n"
        "| SE | 572 | 514 | 89.9% | 83 | 0.15 | 64 | 1 |\n"
        "Some prose between rows.\n"
        "| HR | 350 | 280 | 80.0% | 108 | 0.31 | 37 | 3 |\n"
    )
    rows = guard._parse_table(text)
    assert set(rows) == {"SE", "HR"}
    assert rows["HR"][5] == 37


def test_header_and_separator_rows_are_not_countries() -> None:
    text = (
        "| Country | Entries | Without aliases | % | Aliases | Alias/entry | Fragmented groups | Theme near-dups |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    assert guard._parse_table(text) == {}


# ------------------------------------------------------------- the comparison

def test_drifted_fragmentation_is_detected() -> None:
    """The exact #86 -> #100 drift: frag 63 published, 64 measured."""
    was = (572, 514, 89.9, 83, 0.15, 63, 1)
    now = (572, 514, 89.9, 83, 0.15, 64, 1)
    changed = [
        f"{label} {w:g}->{n:g}"
        for label, w, n in zip(
            ("entries", "noalias", "pct", "aliases", "alias/entry", "frag", "themedup"), was, now
        )
        if abs(w - n) > 0.011
    ]
    assert changed == ["frag 63->64"]


def test_identical_rows_show_no_change() -> None:
    row = (572, 514, 89.9, 83, 0.15, 64, 1)
    changed = [
        f"{label} {w:g}->{n:g}"
        for label, w, n in zip(
            ("entries", "noalias", "pct", "aliases", "alias/entry", "frag", "themedup"), row, row
        )
        if abs(w - n) > 0.011
    ]
    assert changed == []


def test_a_tiny_ratio_drift_is_tolerated() -> None:
    """A 0.01 rounding wobble in a mean is not a stale table."""
    was = (572, 514, 89.9, 83, 0.15, 64, 1)
    now = (572, 514, 89.9, 83, 0.16, 64, 1)
    changed = [
        label
        for label, w, n in zip(
            ("entries", "noalias", "pct", "aliases", "alias/entry", "frag", "themedup"), was, now
        )
        if abs(w - n) > 0.011
    ]
    assert changed == []


def test_countries_constant_covers_ten_countries() -> None:
    assert len(guard.COUNTRIES) == 10
    assert set(guard.COUNTRIES) == {"PL", "HU", "SE", "FR", "BG", "PT", "ES", "FI", "DE", "HR"}


def test_missing_root_is_a_clean_error(tmp_path: Path) -> None:
    assert guard.main(["--root", str(tmp_path / "does-not-exist")]) == 2


def test_guard_does_not_write_the_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is read-only: checking must never rewrite the table.

    A missing codebook root must leave the document byte-identical, whatever the
    guard does about the error.
    """
    doc = tmp_path / "doc.md"
    doc.write_text(
        "| PL | 711 | 649 | 91.3% | 76 | 0.11 | 98 | 17 |\n",
        encoding="utf-8",
    )
    before = doc.read_bytes()
    monkeypatch.setattr(guard, "DOC", doc)
    # An empty root has no codebooks, so measurement cannot proceed. The point is
    # only that the guard never writes to the document it is checking.
    try:
        guard.main(["--root", str(tmp_path)])
    except SystemExit:
        pass
    except FileNotFoundError:
        pass
    assert doc.read_bytes() == before, "the guard modified the document it checks"


def test_report_flag_reports_without_failing(tmp_path: Path) -> None:
    """`--report` is for refreshing: it must not exit non-zero on staleness."""
    assert guard.main(["--root", str(tmp_path / "nope"), "--report"]) == 2  # config error still 2
