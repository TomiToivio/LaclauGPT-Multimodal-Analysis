"""Preflight scoring for the Monday500 pilot resume (#379).

The runbook's HARD STOP conditions were prose. These tests pin them as behaviour: the tool must fail
closed on a wrong dataset, on missing or duplicated pilot identities, and on stale claims — because a
resume submitted against `ep2024_reprocess` is a full-corpus run wearing a pilot's name.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ep24.preflight_pilot import (  # noqa: E402
    PILOT_DATASET,
    CountryReport,
    evaluate,
)


def report(
    country: str = "finland",
    dataset: str = PILOT_DATASET,
    total: int = 500,
    unique_ids: int = 500,
    stages: dict[int, dict[str, int]] | None = None,
    stale_claims: dict[int, int] | None = None,
) -> CountryReport:
    return CountryReport(
        country=country,
        dataset=dataset,
        total=total,
        unique_ids=unique_ids,
        stages=stages or {},
        stale_claims=stale_claims or {},
    )


class TestDatasetIsolation:
    def test_the_pilot_dataset_passes(self) -> None:
        assert evaluate(report(), expected=PILOT_DATASET).ok

    def test_the_full_corpus_namespace_is_refused_by_name(self) -> None:
        verdict = evaluate(report(dataset="ep2024_reprocess"), expected=PILOT_DATASET)
        assert not verdict.ok
        assert "full-corpus namespace" in verdict.problems[0]

    def test_any_other_dataset_is_refused_too(self) -> None:
        verdict = evaluate(report(dataset="ep24_other_pilot"), expected=PILOT_DATASET)
        assert not verdict.ok
        assert "!=" in verdict.problems[0]


class TestIdentityCounts:
    def test_fewer_than_five_hundred_distinct_records_blocks(self) -> None:
        verdict = evaluate(report(total=480, unique_ids=480), expected=PILOT_DATASET)
        assert not verdict.ok
        assert "expected 500" in verdict.problems[0]

    def test_more_distinct_records_block_unless_supersets_are_intended(self) -> None:
        assert not evaluate(report(total=612, unique_ids=612), expected=PILOT_DATASET).ok
        assert evaluate(
            report(total=612, unique_ids=612), expected=PILOT_DATASET, allow_supersets=True
        ).ok

    def test_duplicate_rows_for_the_same_identity_block(self) -> None:
        """`ep24_status.py` shows totals; 500 rows over 480 identities is the trap it cannot show."""
        verdict = evaluate(report(total=520, unique_ids=480), expected=PILOT_DATASET)
        assert not verdict.ok
        assert any("duplicate _storage_id" in problem for problem in verdict.problems)

    def test_the_identity_check_can_be_disabled_for_other_runs(self) -> None:
        assert evaluate(report(total=12, unique_ids=12), expected=PILOT_DATASET, expect_unique=None).ok


class TestClaimState:
    def test_stale_claims_are_reported_per_step(self) -> None:
        verdict = evaluate(
            report(stages={3: {"claimed": 4, "complete": 496}}, stale_claims={3: 4}),
            expected=PILOT_DATASET,
        )
        assert not verdict.ok
        assert "step 3 has 4 stale claim" in verdict.problems[0]

    def test_fresh_claims_are_not_a_problem(self) -> None:
        verdict = evaluate(
            report(stages={3: {"claimed": 4, "complete": 496}}, stale_claims={3: 0}),
            expected=PILOT_DATASET,
        )
        assert verdict.ok

    def test_error_rows_are_visible_in_the_summary_but_not_a_hard_stop(self) -> None:
        """Errors need judgement (retry selectively), so the tool reports them without blocking."""
        verdict = evaluate(report(stages={6: {"error": 3, "complete": 497}}), expected=PILOT_DATASET)
        assert verdict.ok
