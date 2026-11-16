"""The status tool reports identity, because a total is not a count of records (#379).

The issue's runbook warns that `ep24_status.py` shows totals, not uniqueness, and that 500 rows can be
480 records with 20 duplicates — or 500 rows of which two carry no `_storage_id`. These tests hold the
reporting arithmetic to that.

They import the pure module, never the CLI: `ep24_status.py` loads the private environment at import time,
which would set `LACLAUGPT_MONGO_*` for the whole pytest process and break unrelated tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ep24_status_identity import identity_summary  # noqa: E402


class FakeCollection:
    """Only the two reads identity_summary performs."""

    def __init__(self, ids):
        self.ids = list(ids)

    def count_documents(self, query):  # noqa: ARG002 - the fake answers the two shapes we send
        if "_storage_id" in query:
            return sum(1 for value in self.ids if value in (None, ""))
        return len(self.ids)

    def distinct(self, field):
        assert field == "_storage_id"
        seen, out = set(), []
        for value in self.ids:
            if value in (None, "") or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out


def test_a_clean_pilot_reports_equal_total_and_unique():
    summary = identity_summary(FakeCollection([f"id-{n}" for n in range(500)]))
    assert summary == {"total": 500, "unique": 500, "duplicates": 0, "missing_id": 0}


def test_duplicate_rows_are_counted_not_hidden_by_the_total():
    ids = [f"id-{n}" for n in range(500)] + ["id-1", "id-2"]
    summary = identity_summary(FakeCollection(ids))
    assert summary["total"] == 502
    assert summary["unique"] == 500
    assert summary["duplicates"] == 2


def test_rows_without_a_stable_id_are_not_counted_as_duplicates():
    ids = [f"id-{n}" for n in range(498)] + [None, ""]
    summary = identity_summary(FakeCollection(ids))
    assert summary["unique"] == 498
    assert summary["missing_id"] == 2
    assert summary["duplicates"] == 0


def test_an_empty_collection_reports_zeros_rather_than_failing():
    assert identity_summary(FakeCollection([])) == {
        "total": 0, "unique": 0, "duplicates": 0, "missing_id": 0,
    }
