"""EP24 run scope: Steps 1-7 by default, 8-9 paused and opt-in (#346).

The pause is a scope decision, not a deletion: the code, tests and historical outputs stay, and what
changes is that the **default path cannot schedule** SNA or RDF. These tests pin the launcher's
behaviour, because "we remembered not to run 8-9" is not a control.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "scripts" / "roihu" / "run_country_pipeline.sh").read_text(encoding="utf-8")


def test_default_last_step_is_seven() -> None:
    assert re.search(r"^TO_STEP=7$", LAUNCHER, flags=re.MULTILINE)


def test_no_step_nine_default_remains() -> None:
    assert "TO_STEP=9" not in LAUNCHER


def test_steps_eight_and_nine_require_an_explicit_opt_in() -> None:
    assert "LACLAUGPT_EP24_ENABLE_SNA_RDF" in LAUNCHER
    assert "paused (#346)" in LAUNCHER


def test_opt_in_is_checked_before_any_submission() -> None:
    gate = LAUNCHER.index("Steps 8-9 are paused")
    first_submit = LAUNCHER.index("sbatch")
    assert gate < first_submit


def test_help_text_states_the_paused_scope() -> None:
    assert "paused" in LAUNCHER.lower()
    assert "--to-step" in LAUNCHER


def test_step_eight_and_nine_code_is_preserved() -> None:
    """Paused is not deleted: the stage entrypoints must still exist."""
    for name in ("step_8_roihu_social_network_analysis.py", "step_9_roihu_rdf.py"):
        assert (ROOT / name).exists(), f"{name} must not be removed by the pause"
