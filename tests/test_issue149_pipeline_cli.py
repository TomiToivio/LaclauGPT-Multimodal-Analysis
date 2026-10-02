from __future__ import annotations

import os
from pathlib import Path

import pytest

from ep24_cli import checkpoint_path, configure_step_cli, normalize_country


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("finland", "finland"),
        ("FINLAND", "finland"),
        ("fi", "finland"),
        ("poland", "poland"),
        ("PL", "poland"),
        ("portugal", "portugal"),
        ("pt", "portugal"),
    ],
)
def test_country_normalization(value, expected):
    assert normalize_country(value) == expected


def test_invalid_country_fails_clearly():
    with pytest.raises(ValueError, match="unknown country"):
        normalize_country("atlantis")


@pytest.mark.parametrize("step", range(1, 10))
def test_all_steps_accept_country_and_limit(monkeypatch, tmp_path, step):
    monkeypatch.setenv("LACLAUGPT_EP24_INPUT_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setenv("LACLAUGPT_EP24_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.delenv("LACLAUGPT_INPUT_CSV", raising=False)
    monkeypatch.delenv("LACLAUGPT_OUTPUT_CSV", raising=False)
    monkeypatch.delenv("LACLAUGPT_COUNTRY", raising=False)
    monkeypatch.delenv("LACLAUGPT_MAX_ROWS", raising=False)

    selected = configure_step_cli(step, ["--country", "Finland", "--limit", "10"])

    assert selected.country == "finland"
    assert selected.limit == 10
    assert selected.country_source == "cli"
    assert selected.limit_source == "cli"
    assert os.environ["LACLAUGPT_COUNTRY"] == "finland"
    assert os.environ["LACLAUGPT_MAX_ROWS"] == "10"

    if step == 1:
        assert Path(os.environ["LACLAUGPT_INPUT_CSV"]) == tmp_path / "inputs" / "ep24_finland.csv"
    else:
        assert Path(os.environ["LACLAUGPT_INPUT_CSV"]) == checkpoint_path(
            step - 1, "finland", tmp_path / "outputs"
        )
    assert Path(os.environ["LACLAUGPT_OUTPUT_CSV"]) == checkpoint_path(
        step, "finland", tmp_path / "outputs"
    )


def test_cli_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "poland")
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "100")
    monkeypatch.setenv("LACLAUGPT_EP24_OUTPUT_ROOT", str(tmp_path))

    selected = configure_step_cli(3, ["-c", "portugal", "-n", "7"], configure_paths=False)

    assert selected.country == "portugal"
    assert selected.limit == 7
    assert selected.country_source == "cli"
    assert selected.limit_source == "cli"
    assert os.environ["LACLAUGPT_COUNTRY"] == "portugal"
    assert os.environ["LACLAUGPT_MAX_ROWS"] == "7"


def test_environment_only_behavior(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "PL")
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "12")
    selected = configure_step_cli(4, [], configure_paths=False)
    assert selected.country == "poland"
    assert selected.limit == 12
    assert selected.country_source == "environment"
    assert selected.limit_source == "environment"


def test_no_arguments_do_not_invent_global_limit(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_COUNTRY", raising=False)
    monkeypatch.delenv("LACLAUGPT_MAX_ROWS", raising=False)
    monkeypatch.delenv("LACLAUGPT_INPUT_CSV", raising=False)
    monkeypatch.delenv("LACLAUGPT_OUTPUT_CSV", raising=False)

    selected = configure_step_cli(7, [], configure_paths=False)

    assert selected.country is None
    assert selected.limit == 0
    assert selected.country_source == "default"
    assert selected.limit_source == "default"
    # Important: Step 7/8 historically implement their own direct default of 100.
    assert "LACLAUGPT_MAX_ROWS" not in os.environ


@pytest.mark.parametrize("bad", ["-1", "-10"])
def test_negative_limit_rejected(monkeypatch, bad):
    monkeypatch.delenv("LACLAUGPT_MAX_ROWS", raising=False)
    with pytest.raises(ValueError, match="must be >= 0"):
        configure_step_cli(1, ["--limit", bad], configure_paths=False)


def test_zero_limit_is_explicit_unlimited(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_MAX_ROWS", raising=False)
    selected = configure_step_cli(1, ["--limit", "0"], configure_paths=False)
    assert selected.limit == 0
    assert selected.limit_source == "cli"
    assert os.environ["LACLAUGPT_MAX_ROWS"] == "0"


def test_checkpoint_names_preserve_stage_continuity(tmp_path):
    assert checkpoint_path(1, "finland", tmp_path).name == "step_01_preprocess.csv"
    assert checkpoint_path(2, "finland", tmp_path).name == "step_02_frame.csv"
    assert checkpoint_path(3, "poland", tmp_path).name == "step_03_video.csv"
    assert checkpoint_path(8, "portugal", tmp_path).name == "step_08_social_network_analysis.csv"
