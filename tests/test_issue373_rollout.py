"""The rollout controller plans by record and stage, and refuses to run before approval (#373).

The expensive failure this guards against is not a crash: it is paying for 500 videos twice because a
resume was expressed as "skip the first 500 rows", or launching the full corpus because a sample was
*assumed* approved. So the tests are about identity and gating:

* a country that finished Step 5 for 300 of 500 records reports 300 outstanding at Step 6 and nothing at
  Step 5 — the split is per record, not per offset;
* a record another worker has claimed is not outstanding (in flight, not lost, not duplicated);
* a fresh country has every step outstanding and is not "resumable";
* running the plan twice gives the same answer;
* submitting without the approval marker is refused, whatever is in the database.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "ep24"))

import rollout  # noqa: E402


class FakeCollection:
    """A tiny evaluator for the query shapes `eligible_query()` produces."""

    def __init__(self, documents):
        self.documents = documents

    def count_documents(self, query):
        return sum(1 for doc in self.documents if self._matches(doc, query))

    def _matches(self, doc, query):
        for key, expected in query.items():
            if key == "$or":
                if not any(self._matches(doc, branch) for branch in expected):
                    return False
                continue
            value = doc
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(expected, dict):
                if "$in" in expected and value not in expected["$in"]:
                    return False
                if "$exists" in expected:
                    present = value is not None
                    if present is not expected["$exists"]:
                        return False
                if "$nin" in expected and value in expected["$nin"]:
                    return False
            elif value != expected:
                return False
        return True


def _record(country, step_complete=0, *, status=None, claimed_at_step=None):
    doc = {"country": country}
    pipeline = {}
    for step in range(1, step_complete + 1):
        pipeline[f"step_{step:02d}"] = {"status": "complete"}
    if claimed_at_step is not None:
        pipeline[f"step_{claimed_at_step:02d}"] = {"status": "claimed", "claimed_at": "2026-11-04T00:00:00Z"}
    doc["_pipeline"] = pipeline
    doc["preprocess_status"] = "ok"
    doc["frame_file"] = "x.jpg"
    doc["frame_timestamp_seconds"] = 1.0
    doc["frame_analysis_status"] = "ok"
    doc["processing_status"] = "OK"
    return doc


def test_a_half_finished_country_resumes_per_record_not_per_offset():
    docs = [_record("fi", step_complete=5) for _ in range(300)] + [_record("fi") for _ in range(200)]
    plan = rollout.plan_country(FakeCollection(docs), "fi")
    assert plan["records"] == 500
    # The 200 that never started are eligible at Step 1 and at no later step, because Step 5 requires
    # Step 4 complete: "outstanding" means "claimable now", which is what the runner would actually do.
    assert plan["steps"]["1"]["outstanding"] == 200
    assert plan["steps"]["5"]["outstanding"] == 0
    assert plan["steps"]["5"]["complete"] == 300
    assert plan["steps"]["6"]["outstanding"] == 300   # the 300 that did finish Step 5
    assert plan["resumable"] is True


def test_a_finished_country_has_nothing_outstanding():
    docs = [_record("fi", step_complete=9) for _ in range(500)]
    plan = rollout.plan_country(FakeCollection(docs), "fi")
    assert plan["outstanding"] == 0
    assert plan["complete_through_final_step"] is True


def test_a_fresh_country_starts_everywhere_and_is_not_resumable():
    docs = [_record("de") for _ in range(40)]
    plan = rollout.plan_country(FakeCollection(docs), "de")
    assert plan["steps"]["1"]["outstanding"] == 40
    assert plan["resumable"] is False


def test_a_claimed_record_is_not_outstanding():
    docs = [_record("pt", step_complete=4, claimed_at_step=5)]
    plan = rollout.plan_country(FakeCollection(docs), "pt")
    assert plan["steps"]["5"]["outstanding"] == 0, "in-flight work must not be queued a second time"


def test_the_plan_is_deterministic():
    docs = [_record("pl", step_complete=2) for _ in range(5)]
    first = rollout.build_plan(FakeCollection(docs), ["pl"])
    second = rollout.build_plan(FakeCollection(docs), ["pl"])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_discovery_excludes_the_sample_countries_and_orders_the_rest():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for country in ("fi", "pl", "pt", "se", "de", "hu"):
            (Path(tmp) / f"ep24_{country}.csv").write_text("id\n")
        assert rollout.remaining_countries(Path(tmp)) == ["de", "hu", "se"]


def test_submitting_without_approval_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("EP24_SAMPLE_APPROVED", raising=False)
    monkeypatch.setattr(rollout, "_collection_from_settings",
                        lambda: FakeCollection([_record("fi", step_complete=3)]))
    called = []
    monkeypatch.setattr(rollout, "submit_country", lambda *a, **k: called.append(a) or 0)

    assert rollout.main(["--mode", "resume-sample", "--submit"]) == 3
    assert called == []
    assert "not approved" in capsys.readouterr().err


def test_approval_lets_the_submission_through(monkeypatch):
    monkeypatch.setenv("EP24_SAMPLE_APPROVED", "1")
    monkeypatch.setattr(rollout, "_collection_from_settings",
                        lambda: FakeCollection([_record("fi", step_complete=3)]))
    called = []
    monkeypatch.setattr(rollout, "submit_country", lambda country, **k: called.append(country) or 0)

    assert rollout.main(["--mode", "resume-sample", "--submit", "--country", "fi"]) == 0
    assert called == ["fi"]


def test_a_finished_country_is_not_submitted(monkeypatch):
    monkeypatch.setenv("EP24_SAMPLE_APPROVED", "1")
    monkeypatch.setattr(rollout, "_collection_from_settings",
                        lambda: FakeCollection([_record("fi", step_complete=9)]))
    called = []
    monkeypatch.setattr(rollout, "submit_country", lambda country, **k: called.append(country) or 0)

    assert rollout.main(["--mode", "resume-sample", "--submit", "--country", "fi"]) == 0
    assert called == []


def test_dry_run_is_the_default(monkeypatch, capsys):
    monkeypatch.delenv("EP24_SAMPLE_APPROVED", raising=False)
    monkeypatch.setattr(rollout, "_collection_from_settings",
                        lambda: FakeCollection([_record("fi", step_complete=2)]))
    assert rollout.main(["--mode", "resume-sample", "--country", "fi"]) == 0
    out = capsys.readouterr().out
    assert "cells are outstanding records per step" in out


def test_step_seven_completion_still_needs_sna_but_pilot_can_stop_at_seven():
    collection = FakeCollection([_record("fi", step_complete=7)])
    full = rollout.plan_country(collection, "fi")
    assert full["steps"]["8"]["outstanding"] == 1
    assert not full["complete_through_final_step"]
    pilot = rollout.plan_country(collection, "fi", steps=tuple(range(1, 8)))
    assert pilot["outstanding"] == 0
    assert pilot["complete_through_final_step"]
