"""#380: SNA and RDF are active steps again."""
from pathlib import Path
import ep24_scope


def test_normal_run_includes_all_stages():
    assert ep24_scope.ACTIVE_STEPS == tuple(range(10))
    assert ep24_scope.PAUSED_STEPS == ()
    assert ep24_scope.default_last_step() == 9


def test_steps_eight_and_nine_active_without_opt_in(monkeypatch):
    monkeypatch.delenv(ep24_scope.INCLUDE_PAUSED_ENV, raising=False)
    for step in range(10):
        assert ep24_scope.allows(step)
        assert not ep24_scope.is_paused(step)
        assert ep24_scope.scope_label(step) == "active"
        ep24_scope.guard(step)


def test_country_launcher_includes_sna_and_rdf():
    script = (Path(__file__).resolve().parents[1] / "scripts/roihu/run_country_pipeline.sh").read_text()
    assert "TO_STEP=9" in script
    assert "EP24_INCLUDE_PAUSED_STEPS" not in script
