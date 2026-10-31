"""One log directory and one filename convention (#324).

The behaviour under test is the contract a human relies on when something breaks at 02:00: there is one
directory, the file is named after the step and the country, and a rerun does not destroy the previous
run's log.
"""
from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ep24_logging as el  # noqa: E402

COUNTRIES = ("finland", "poland", "portugal")
STEPS = tuple(el.STEP_NAMES.items())


def test_every_step_and_country_produces_the_canonical_filename(tmp_path: Path) -> None:
    for number, name in STEPS:
        for country in COUNTRIES:
            path = el.canonical_log_path(number, country, root=tmp_path)
            assert path.name == f"{name}-{country}.log"
            assert path.parent == tmp_path  # never a per-step or per-country subdirectory


@pytest.mark.parametrize(
    "given,expected",
    [(1, "preprocess"), ("preprocess", "preprocess"), ("PREPROCESS", "preprocess"),
     ("video", "video_analysis"), ("sna", "social_network"), ("9", "rdf")],
)
def test_step_normalization(given, expected) -> None:
    assert el.normalize_step(given) == expected


def test_unknown_step_raises_instead_of_inventing_a_name() -> None:
    with pytest.raises(ValueError):
        el.normalize_step(42)
    with pytest.raises(ValueError):
        el.normalize_step("preproces")  # a typo must fail, not create a near-miss file


@pytest.mark.parametrize(
    "given,expected",
    [("FI", "finland"), ("fi", "finland"), (" finland ", "finland"),
     ("PL", "poland"), ("pt", "portugal"), ("all", "all")],
)
def test_country_normalization(given, expected) -> None:
    assert el.normalize_country(given) == expected


def test_unknown_or_missing_country_raises() -> None:
    with pytest.raises(ValueError):
        el.normalize_country("brazil")
    with pytest.raises(ValueError):
        el.normalize_country(None)
    with pytest.raises(ValueError):
        el.normalize_country("   ")


def test_primary_env_var_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EP24_LOG_DIR", str(tmp_path / "primary"))
    monkeypatch.setenv("LACLAUGPT_EP24_LOG_DIR", str(tmp_path / "legacy"))
    assert el.log_dir() == tmp_path / "primary"


def test_legacy_env_var_is_still_honoured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EP24_LOG_DIR", raising=False)
    monkeypatch.setenv("LACLAUGPT_EP24_LOG_DIR", str(tmp_path / "legacy"))
    assert el.log_dir() == tmp_path / "legacy"


def test_default_is_the_private_root_not_the_repository(monkeypatch, tmp_path: Path) -> None:
    for name in el.ROOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", str(tmp_path / "private"))
    assert el.log_dir() == tmp_path / "private" / "logs"


def test_directory_is_created_with_owner_only_permissions(tmp_path: Path) -> None:
    path = el.log_dir(tmp_path / "made")
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_oversized_log_is_archived_and_the_canonical_name_freed(tmp_path: Path) -> None:
    path = el.canonical_log_path(1, "poland", root=tmp_path)
    path.write_text("x" * 200)
    archived = el._rotate_if_large(path, max_bytes=100)
    assert archived is not None and archived.exists()
    assert not path.exists()
    assert archived.read_text() == "x" * 200


def test_configure_appends_rather_than_truncating(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EP24_LOG_DIR", str(tmp_path))
    first = el.configure(2, "finland")
    logging.getLogger("test").info("first run")
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()
    second = el.configure(2, "finland")
    logging.getLogger("test").info("second run")
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()
    assert first == second
    content = second.read_text()
    assert "first run" in content and "second run" in content


def test_slurm_logs_share_the_directory_but_not_the_file(tmp_path: Path) -> None:
    out = el.slurm_log_path(9, "portugal", "12345", "out", root=tmp_path)
    err = el.slurm_log_path(9, "portugal", "12345", "err", root=tmp_path)
    python_log = el.canonical_log_path(9, "portugal", root=tmp_path)
    assert out.name == "rdf-portugal-12345.out"
    assert err.name == "rdf-portugal-12345.err"
    assert out.parent == python_log.parent == tmp_path
    assert out != python_log
    with pytest.raises(ValueError):
        el.slurm_log_path(9, "portugal", "1", "stdout", root=tmp_path)


def test_country_agnostic_steps_have_an_honest_name(tmp_path: Path) -> None:
    assert el.canonical_log_path(0, "all", root=tmp_path).name == "import-all.log"


def test_configure_from_env_uses_the_submitter_exports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EP24_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LACLAUGPT_STEP", "7")
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "pt")
    path = el.configure_from_env()
    assert path.name == "discourse_network-portugal.log"
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()
