"""Regression contract: the active EP24 country chain stops at Step 7 (#346)."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/roihu/run_country_pipeline.sh"


def test_default_scope_pauses_sna_and_rdf():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TO_STEP=7" in text
    assert "ALLOW_PAUSED=0" in text
    assert "afterany:${PREVIOUS}" in text
    assert "Steps 8 (SNA) and 9 (RDF) are paused" in text


def test_paused_steps_cannot_be_submitted_accidentally():
    # Validation occurs before sbatch availability: no Roihu credentials required.
    result = subprocess.run(
        ["bash", str(SCRIPT), "finland", "--to-step", "9"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "paused" in result.stderr
    assert "sbatch" not in result.stderr


def test_explicit_opt_in_is_available_but_not_default():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--enable-paused-stages) ALLOW_PAUSED=1" in text
    assert "if (( (FROM_STEP > 7 || TO_STEP > 7) && ! ALLOW_PAUSED )); then" in text
