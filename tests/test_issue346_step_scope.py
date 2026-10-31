"""#346: Steps 8-9 are paused by default, and that must be visible rather than implicit."""
from __future__ import annotations

import os

import pytest

import ep24_scope


def test_a_normal_run_ends_at_seven():
    assert ep24_scope.default_last_step() == 7
    assert ep24_scope.ACTIVE_STEPS == (0, 1, 2, 3, 4, 5, 6, 7)


def test_paused_steps_are_named_not_deleted():
    assert ep24_scope.PAUSED_STEPS == (8, 9)
    assert ep24_scope.is_paused(8) and ep24_scope.is_paused(9)


def test_steps_one_to_seven_are_always_allowed():
    for step in range(0, 8):
        assert ep24_scope.allows(step)


def test_paused_steps_are_refused_without_opt_in(monkeypatch):
    monkeypatch.delenv(ep24_scope.INCLUDE_PAUSED_ENV, raising=False)
    assert not ep24_scope.allows(8)
    with pytest.raises(ValueError) as exc:
        ep24_scope.guard(9)
    assert "paused" in str(exc.value)
    assert ep24_scope.INCLUDE_PAUSED_ENV in str(exc.value), "the message must say how to opt in"


def test_explicit_opt_in_re_enables_them(monkeypatch):
    monkeypatch.setenv(ep24_scope.INCLUDE_PAUSED_ENV, "1")
    assert ep24_scope.paused_included()
    assert ep24_scope.allows(8) and ep24_scope.allows(9)
    ep24_scope.guard(9)  # must not raise


def test_the_opt_in_needs_a_truthy_value(monkeypatch):
    for value in ("0", "", "no", "false"):
        monkeypatch.setenv(ep24_scope.INCLUDE_PAUSED_ENV, value)
        assert not ep24_scope.allows(9), value


def test_labels_distinguish_paused_from_active():
    assert ep24_scope.scope_label(6) == "active"
    assert ep24_scope.scope_label(8) == "paused"


def test_the_launcher_default_stops_at_seven():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts/roihu/run_country_pipeline.sh").read_text()
    assert "TO_STEP=7" in script, "the default run must end at Step 7"
    assert "EP24_INCLUDE_PAUSED_STEPS" in script, "the opt-in must be explicit"
    assert "exit 2" in script
