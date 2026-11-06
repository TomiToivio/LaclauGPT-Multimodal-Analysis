"""The #379 hard stop: a total is not a count of records.

The runbook says "verify each country reports the expected 500 unique pilot records (the simple status
command shows total, not uniqueness)". These tests hold the status tool to that: distinct stable ids are
reported, duplicates and id-less rows fail the check, and ``--expect`` is enforced by the same function
the command uses rather than by an operator's judgement.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_status_module():
    path = ROOT / "scripts" / "ep24" / "ep24_status.py"
    spec = importlib.util.spec_from_file_location("ep24_status_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_clean_pilot_passes_the_hard_stop():
    status = load_status_module()
    summary = status.identity_summary(FakeCollection([f"id-{n}" for n in range(500)]))
    assert summary == {"total": 500, "unique": 500, "duplicates": 0, "missing_id": 0}
    assert status.identity_failures("finland", summary, 500) == []


def test_duplicate_rows_are_counted_not_hidden_by_the_total():
    status = load_status_module()
    ids = [f"id-{n}" for n in range(500)] + ["id-1", "id-2"]
    summary = status.identity_summary(FakeCollection(ids))
    assert summary["total"] == 502
    assert summary["unique"] == 500
    assert summary["duplicates"] == 2
    failures = status.identity_failures("poland", summary, 500)
    assert any("duplicate" in line for line in failures)


def test_rows_without_a_stable_id_cannot_pass():
    status = load_status_module()
    ids = [f"id-{n}" for n in range(498)] + [None, ""]
    summary = status.identity_summary(FakeCollection(ids))
    assert summary["unique"] == 498
    assert summary["missing_id"] == 2
    assert summary["duplicates"] == 0
    assert any("without a stable id" in line for line in status.identity_failures("portugal", summary, 500))


def test_expect_mismatch_names_the_numbers():
    status = load_status_module()
    summary = {"total": 500, "unique": 480, "duplicates": 20, "missing_id": 0}
    failures = status.identity_failures("finland", summary, 500)
    assert any("expected 500 distinct ids, found 480" in line for line in failures)


def test_no_expectation_still_reports_identity_problems():
    status = load_status_module()
    summary = {"total": 501, "unique": 500, "duplicates": 1, "missing_id": 0}
    assert status.identity_failures("finland", summary, None)
