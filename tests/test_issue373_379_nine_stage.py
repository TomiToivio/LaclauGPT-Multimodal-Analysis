"""Regression guards for the all-nine-stage EP24 rollout (#373, #379).

These static checks catch the November regression where the country runner
supported nine stages but the approval-gated full-corpus wrapper ran only seven.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_corpus_launcher_is_nine_stage_and_dataset_guarded():
    code = (ROOT / "scripts/ep24/run_ep24_rollout.sh").read_text()
    assert "--from-step 1 --to-step 9 --require-dataset ep2024_reprocess" in code
    assert "Submitting stages 1-9" in code
    assert "squeue" in code
    assert "--approval-file" in code
    assert "--to-step 7" not in code


def test_pilot_preflight_inspects_steps_one_through_nine():
    code = (ROOT / "scripts/ep24/preflight_pilot.py").read_text()
    assert "for step in range(1, 10):" in code
    assert 'PILOT_DATASET = "ep24_monday500_20261031"' in code
