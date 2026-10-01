"""Restartability tests for the EP24 stage state machine (issue #64).

These cover the rules the issue calls essential, and they run on CI without a
live MongoDB or Redis -- the state service is storage-agnostic and these tests
use the in-memory store.

The scenario the issue describes explicitly, and which is tested here:
run Step 1 for a while, stop it, submit Step 2, and have Step 2 process exactly
the rows Step 1 finished; then resume Step 1 and have Step 2 pick up the newly
available rows on its next run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_stage_state as st  # noqa: E402


def _service(stage, *, clock=None, ttl=3600.0, owner="test"):
    store = st.MemoryStateStore()
    kwargs = {"claim_ttl_seconds": ttl, "owner": owner}
    if clock is not None:
        kwargs["now"] = clock
    return st.StageStateService(stage=stage, store=store, **kwargs)


# --- stage N+1 requires stage N complete ----------------------------------

def test_stage_two_processes_nothing_until_stage_one_completes():
    """The dependency gate: no prior stage complete -> nothing eligible."""
    one, two = _service(1), _service(2)
    ids = ["r1", "r2"]

    assert two.eligible(ids, previous=one) == []

    one.complete("r1")
    assert two.eligible(ids, previous=one) == ["r1"]


def test_stage_one_has_no_predecessor_and_is_eligible_immediately():
    one = _service(1)
    assert one.eligible(["r1", "r2"]) == ["r1", "r2"]


# --- the partial-run workflow the issue describes -------------------------

def test_partial_step_one_then_step_two_then_resume_picks_up_new_rows():
    one, two = _service(1), _service(2)
    ids = ["r1", "r2", "r3", "r4"]

    # Step 1 runs for a while and completes only half the rows.
    for record_id in one.claim(["r1", "r2"]):
        one.complete(record_id)

    # Step 2 can already run over exactly that subset.
    assert two.eligible(ids, previous=one) == ["r1", "r2"]
    for record_id in two.claim(two.eligible(ids, previous=one)):
        two.complete(record_id)
    assert two.eligible(ids, previous=one) == [], "the finished subset is not reprocessed"

    # Step 1 is resumed and finishes the rest.
    for record_id in one.claim(["r3", "r4"]):
        one.complete(record_id)

    # Step 2's next run picks up only the newly available rows.
    assert two.eligible(ids, previous=one) == ["r3", "r4"]

    for record_id in two.claim(two.eligible(ids, previous=one)):
        two.complete(record_id)
    assert two.eligible(ids, previous=one) == []


def test_completed_rows_are_never_reprocessed_without_force():
    one = _service(1)
    one.complete("r1")
    assert one.eligible(["r1"]) == []
    assert one.eligible(["r1"], force=True) == ["r1"]


def test_rerun_does_not_duplicate_successful_work():
    """Idempotence: claiming an already-complete record changes nothing."""
    one = _service(1)
    one.complete("r1")
    assert one.claim(["r1"]) == []
    assert one.counts()["complete"] == 1


# --- claims, stale recovery, concurrency ----------------------------------

def test_a_live_claim_is_not_handed_to_a_second_worker():
    now = [1000.0]
    a = _service(1, clock=lambda: now[0], ttl=60, owner="worker-a")
    b = _service(1, clock=lambda: now[0], ttl=60, owner="worker-b")
    b.store = a.store  # same database

    assert a.claim(["r1"]) == ["r1"]
    assert b.claim(["r1"]) == [], "a live claim must not be double-claimed"


def test_a_stale_claim_is_recoverable_after_the_ttl():
    """A worker that dies mid-batch must not strand its rows forever."""
    now = [1000.0]
    a = _service(1, clock=lambda: now[0], ttl=60)
    b = _service(1, clock=lambda: now[0], ttl=60, owner="worker-b")
    b.store = a.store

    a.claim(["r1"])
    assert b.reclaim_stale() == [], "not stale yet"

    now[0] += 61
    assert b.reclaim_stale() == ["r1"]
    assert b.claim(["r1"]) == ["r1"], "the row is workable again after recovery"


def test_errors_are_claimable_again_and_count_attempts():
    one = _service(1)
    one.claim(["r1"])
    one.fail("r1", "boom")
    assert one.counts()["error"] == 1

    assert one.eligible(["r1"]) == ["r1"], "an errored row can be retried"
    one.claim(["r1"])
    one.fail("r1", "boom again")
    record = one.store.load(1)["r1"]
    assert record.attempts == 2


def test_retry_moves_an_error_back_to_a_claimable_state():
    one = _service(1)
    one.fail("r1", "boom")
    one.retry("r1")
    assert one.eligible(["r1"]) == ["r1"]


def test_skipped_records_are_terminal_but_distinguishable_from_complete():
    one = _service(1)
    one.skip("r1", "no media")
    assert one.eligible(["r1"]) == []
    assert one.counts()["skipped"] == 1
    assert one.counts()["complete"] == 0


def test_complete_clears_the_claim_and_records_provenance():
    one = _service(1)
    one.claim(["r1"])
    record = one.complete("r1", provenance={"model": "synthetic-model", "prompt_sha256": "abc"})
    assert record.claimed_at is None
    assert record.claimed_by is None
    assert record.completed_at is not None
    assert record.provenance["model"] == "synthetic-model"


def test_completing_an_unknown_record_does_not_crash():
    """A record with no prior state is created rather than raising."""
    one = _service(1)
    record = one.complete("never-seen")
    assert record.state == "complete"


def test_counts_reports_every_state_key():
    one = _service(1)
    one.complete("r1")
    one.fail("r2", "err")
    one.skip("r3", "skip")
    one.claim(["r4"])
    counts = one.counts()
    for state in st.StageState:
        assert state.value in counts
    assert counts["complete"] == 1 and counts["error"] == 1
    assert counts["skipped"] == 1 and counts["claimed"] == 1
