"""Static regression checks for Step 9 Roihu launcher and prerequisites."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step9_skips_architecture_incompatible_allas_module():
    profile = (ROOT / "scripts/roihu/roihu_step_profiles.sh").read_text()
    assert 'if [[ "${step}" != "9" ]]; then' in profile
    assert 'module load allas' in profile


def test_run_step_places_logs_in_private_workspace():
    runner = (ROOT / "scripts/roihu/run_step.sh").read_text()
    assert 'LACLAUGPT_EP24_LOG_ROOT' in runner
    assert '${PRIVATE_ROOT}/logs' in runner
    assert '--output="${LOG_ROOT}/ep24_s${STEP}_%j.out"' in runner
    assert '--error="${LOG_ROOT}/ep24_s${STEP}_%j.err"' in runner


def test_step9_without_upstream_fails_with_actionable_diagnostic():
    orchestrator = (ROOT / "ep24_stage_orchestrator.py").read_text()
    assert 'if step == 9 and upstream == 0:' in orchestrator
    assert 'zero Step 8' in orchestrator
    assert 'cannot generate missing Step 8 results' in orchestrator
