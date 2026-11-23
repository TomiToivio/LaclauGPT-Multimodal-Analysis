"""The shared stage helpers have one home, and behave as the copies did (#413).

`clean_document` and `bounded_text` were byte-identical copies inside `roihu_discourse_network.py` and
`roihu_populism.py`. These tests pin the moved behaviour so the consolidation cannot silently change what
the stages write into the cumulative documents.

All fixtures are synthetic (AGENTS.md public/private boundary).
"""

from __future__ import annotations

import math

import pandas as pd

from roihu_frames import bounded_text, clean_document


def test_clean_document_turns_missing_values_into_none():
    row = pd.Series({"a": 1, "b": None, "c": float("nan")})
    doc = clean_document(row)
    assert doc["a"] == 1
    assert doc["b"] is None
    assert doc["c"] is None


def test_clean_document_coerces_keys_and_preserves_unusual_values():
    row = pd.Series({"1": "x"})
    row.index = [1]
    doc = clean_document(row)
    assert list(doc) == ["1"]
    assert doc["1"] == "x"


def test_clean_document_keeps_values_that_cannot_be_null_tested():
    # A list has no scalar nullness; pd.isna returns an array, which must not drop the value.
    row = pd.Series({"tags": [["a", "b"]]})
    doc = clean_document(row)
    assert doc["tags"] == [["a", "b"]]


def test_clean_document_passes_infinities_through():
    # Infinities are not missing values and must survive, exactly as before the move.
    doc = clean_document(pd.Series({"x": math.inf}))
    assert doc["x"] == math.inf


def test_bounded_text_reports_truncation_without_appending_anything():
    assert bounded_text("abc", 5) == ("abc", False)
    assert bounded_text("abcdef", 3) == ("abc", True)
    assert bounded_text("", 0) == ("", False)
