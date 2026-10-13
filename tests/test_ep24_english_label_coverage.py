"""Synthetic regression tests for issue #101 bilingual-label QA."""
import json
from pathlib import Path

import pytest

from roihu_codebooks import english_label_coverage, load_profile


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    books = tmp_path / "codebooks"
    books.mkdir()
    _write(books / "ep24_common_private.json", {"country_code": "COMMON", "entries": []})
    _write(
        books / "ep24_finland_private.json",
        {
            "country_code": "FI",
            "language": "fi",
            "entries": [
                {"id": "translated", "kind": "topic", "label": "ilmasto", "english_label": "climate"},
                {"id": "same-name", "kind": "actor", "label": "Ada Lovelace", "english_label": "Ada Lovelace"},
                {"id": "missing", "kind": "topic", "label": "demokratia"},
                {
                    "id": "exempt",
                    "kind": "entity",
                    "label": "X-42",
                    "metadata": {"english_label_exempt_reason": "language-neutral identifier"},
                },
            ],
        },
    )
    return tmp_path


def test_english_label_coverage_is_actionable(tmp_path: Path) -> None:
    entries, meta = load_profile(_fixture(tmp_path), "FI")
    qa = meta["english_label_coverage"]
    assert qa["required_count"] == 3
    assert qa["present_count"] == 2
    assert qa["missing_count"] == 1
    assert qa["exempt_count"] == 1
    assert qa["state"] == "REVIEW_REQUIRED"
    assert meta["qa_state"] == "REVIEW_REQUIRED"
    assert meta["missing_english_entry_ids"] == ["missing"]


def test_identical_english_name_must_still_be_explicit(tmp_path: Path) -> None:
    entries, _ = load_profile(_fixture(tmp_path), "FI")
    qa = english_label_coverage(entries)
    assert "same-name" not in qa["missing_entry_ids"]


def test_strict_english_gate_rejects_missing_required_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires English-label review"):
        load_profile(_fixture(tmp_path), "FI", strict_english=True)


def test_exemption_requires_documented_reason(tmp_path: Path) -> None:
    entries, _ = load_profile(_fixture(tmp_path), "FI")
    exempt = next(e for e in entries if e.entry_id == "exempt")
    assert exempt.metadata["english_label_exempt_reason"]
