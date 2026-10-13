"""Tests for the EP24 private-settings module used by the Roihu bootstrap (#188).

The point of this module is that a *public* pipeline can read *private*
configuration without any secret reaching the public repository, a Slurm log, or
a test failure message. The redaction cases are therefore the important ones,
not the parsing ones.

`ep24_settings.py` predates this issue and is used by the pipeline steps
(`scripts/ep24/ep24_status.py`, `scripts/ep24/bootstrap_ep24_mongodb.py`), so its
existing `private_root()` / `load_private_env()` contract is exercised here too:
the #188 additions must not break it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ep24_settings as S  # noqa: E402

SECRET = "SUPERSECRET-do-not-leak-42"


@pytest.fixture(autouse=True)
def _isolate_laclauGPT_env():
    """Snapshot and restore the whole ``LACLAUGPT_*`` environment per test.

    ``load_private_env`` writes into the process-global ``os.environ`` with
    ``setdefault``. Without this, a value loaded by one test stays set for every
    later test in the same pytest process -- which silently broke five unrelated
    pipeline tests the first time this file was added.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("LACLAUGPT_")}
    for name in [n for n in os.environ if n.startswith("LACLAUGPT_")]:
        os.environ.pop(name, None)
    yield
    for name in [n for n in os.environ if n.startswith("LACLAUGPT_")]:
        os.environ.pop(name, None)
    os.environ.update(saved)


@pytest.fixture()
def private(tmp_path):
    """A private-checkout-like tree with a .env. Isolation is autouse above."""
    root = tmp_path / "LaclauGPT-Private"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_env(root: Path, text: str) -> Path:
    (root / ".env").write_text(text, encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# redaction: the reason these helpers exist
# --------------------------------------------------------------------------- #

def test_summary_never_prints_a_secret(private, monkeypatch):
    _write_env(private, f"LACLAUGPT_MONGO_URI=mongodb://user:{SECRET}@host:27017/db\n")
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    S.load_private_env()
    out = S.summary()
    assert SECRET not in out, "a credential must never be printable"
    assert "redacted" in out


def test_describe_redacts_only_secret_shaped_names(private, monkeypatch):
    env = {
        "LACLAUGPT_MONGO_URI": "mongodb://x",
        "LACLAUGPT_API_TOKEN": "abc123",
        "LACLAUGPT_MAX_ROWS": "100",
        "PATH": "/usr/bin",
    }
    joined = "\n".join(S.describe(env))
    assert "mongodb://x" not in joined
    assert "abc123" not in joined
    assert "LACLAUGPT_MAX_ROWS=100" in joined
    # describes only LACLAUGPT_* settings, not the whole environment
    assert "PATH" not in joined


def test_secret_marker_matching_covers_the_names_the_pipeline_uses():
    for name in ("LACLAUGPT_MONGO_URI", "LACLAUGPT_API_TOKEN", "LACLAUGPT_PASSWORD",
                 "LACLAUGPT_SECRET", "SOME_CONNECTION_STRING"):
        assert S.is_secret(name), name
    for name in ("LACLAUGPT_MAX_ROWS", "LACLAUGPT_EP24_ROOT", "LACLAUGPT_COUNTRY"):
        assert not S.is_secret(name), name


def test_redact_hides_length_class_but_not_the_value():
    out = S.redact("abcdef")
    assert "abcdef" not in out
    assert "6" in out
    assert S.redact("") == "<empty>"


# --------------------------------------------------------------------------- #
# the pre-existing load_private_env contract must keep working
# --------------------------------------------------------------------------- #

def test_load_private_env_applies_the_file_without_overriding_the_environment(
    private, monkeypatch
):
    _write_env(private, "LACLAUGPT_MAX_ROWS=100\nLACLAUGPT_FROM_FILE=yes\n")
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "5")  # a Slurm export must win
    S.load_private_env()
    import os
    assert os.environ["LACLAUGPT_MAX_ROWS"] == "5", "the environment has precedence"
    assert os.environ["LACLAUGPT_FROM_FILE"] == "yes"


def test_load_private_env_returns_the_path_it_read(private, monkeypatch):
    _write_env(private, "LACLAUGPT_A=1\n")
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    # It returns the settings FILE it read, not the root directory.
    assert S.load_private_env() == private / ".env"


def test_load_private_env_survives_a_missing_file(private, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    returned = S.load_private_env()  # no .env written
    assert not returned.exists()
    import os
    # the documented defaults are still applied
    assert os.environ["LACLAUGPT_DATASET"] == "ep2024_reprocess"


def test_private_root_prefers_the_ep24_variable(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", "/a/ep24")
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "/b/multimodal")
    assert str(S.private_root()) == "/a/ep24"
    monkeypatch.delenv("LACLAUGPT_EP24_PRIVATE_ROOT")
    assert str(S.private_root()) == "/b/multimodal"


def test_parsing_handles_export_quotes_and_malformed_lines(private, monkeypatch):
    _write_env(
        private,
        "# comment\n\nexport LACLAUGPT_A=1\n"
        'LACLAUGPT_B="two words"\nLACLAUGPT_C=\'three words\'\n'
        "garbage line without equals\n",
    )
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    S.load_private_env()
    import os
    assert os.environ["LACLAUGPT_A"] == "1"
    assert os.environ["LACLAUGPT_B"] == "two words"
    assert os.environ["LACLAUGPT_C"] == "three words"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_validate_reports_every_missing_name_at_once():
    missing = S.validate({}, ("LACLAUGPT_MONGO_URI", "LACLAUGPT_OTHER"))
    assert missing == ["LACLAUGPT_MONGO_URI", "LACLAUGPT_OTHER"]
    assert S.validate({"LACLAUGPT_MONGO_URI": "x"}, ("LACLAUGPT_MONGO_URI",)) == []


def test_summary_reports_missing_required(private, monkeypatch):
    _write_env(private, "LACLAUGPT_MAX_ROWS=1\n")
    monkeypatch.setenv("LACLAUGPT_EP24_PRIVATE_ROOT", str(private))
    S.load_private_env()
    out = S.summary()
    assert "LACLAUGPT_MONGO_URI" in out
    assert "missing_required" in out


# --------------------------------------------------------------------------- #
# the bootstrap CLI
# --------------------------------------------------------------------------- #

def test_cli_exit_code_reflects_missing_settings(tmp_path, monkeypatch, capsys):
    """Exit 1 when a required setting is absent, 0 when present.

    ``os.environ`` is process-global and ``load_private_env`` uses setdefault, so
    a value loaded for one root stays set. Each Slurm job is a fresh process, so
    the test clears the setting between the two calls to model that.
    """
    good = tmp_path / "good"
    good.mkdir()
    _write_env(good, "LACLAUGPT_MONGO_URI=mongodb://h/db\n")
    assert S.main(["--private-root", str(good)]) == 0

    monkeypatch.delenv("LACLAUGPT_MONGO_URI", raising=False)
    bad = tmp_path / "bad"
    bad.mkdir()
    _write_env(bad, "LACLAUGPT_MAX_ROWS=1\n")
    assert S.main(["--private-root", str(bad)]) == 1


def test_cli_output_leaks_no_secret(tmp_path, capsys):
    root = tmp_path / "priv"
    root.mkdir()
    _write_env(root, f"LACLAUGPT_MONGO_URI=mongodb://u:{SECRET}@h/db\n")
    S.main(["--private-root", str(root)])
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
