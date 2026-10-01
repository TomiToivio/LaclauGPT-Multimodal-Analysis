"""Bootstrap CLI tests (issue #64).

The CLI is the operator's one-time entry point, so the properties worth pinning
are: it discovers the country inputs in the documented priority order, it refuses
to guess when the private inputs are absent, it never writes the private source,
and a dry run really writes nothing.

Fixtures here are synthetic CSVs written into tmp_path. No private research data
is used.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLI = ROOT / "scripts" / "ep24" / "bootstrap_ep24_mongodb.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("bootstrap_cli", CLI)
    assert spec is not None and spec.loader is not None, f"cannot load {CLI}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADER = [
    "video_id", "country", "author_username", "account_type", "source_type",
    "source_recording", "sequence_number", "political_preference", "allas_filename",
    "new_entity", "new_theme", "video_duration", "researcher_new_persons",
    "researcher_new_themes", "researcher_note",
]


def _write_input(root: Path, token: str, *, rows: int = 2) -> Path:
    directory = root / "analysis" / "ep24_reprocess" / "data" / "to_reprocess"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ep24_{token}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for index in range(rows):
            writer.writerow([
                f"SYNTH-{token.upper()}-{index}", token.title(), "synthetic-profile-a",
                "Synthetic", "Instagram", "screen-synth", str(index), "centre right",
                "https://example.invalid/s.mp4", "", "synthetic theme", "10.0",
                "[]", "[]", "",
            ])
    return path


@pytest.fixture()
def cli():
    return _load_cli()


def test_locate_inputs_fails_loudly_when_the_private_root_is_wrong(cli, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cli.discover_inputs(tmp_path)
    message = str(excinfo.value)
    assert "to_reprocess" in message
    assert "private" in message.lower()


def test_discover_inputs_orders_the_demo_countries_first(cli, tmp_path):
    for token in ("sweden", "portugal", "finland", "poland"):
        _write_input(tmp_path, token)
    found = cli.discover_inputs(tmp_path)
    assert list(found)[:3] == ["Finland", "Poland", "Portugal"]
    assert list(found)[3:] == ["Sweden"]


def test_discover_inputs_ignores_an_unknown_country_token(cli, tmp_path):
    _write_input(tmp_path, "finland")
    _write_input(tmp_path, "atlantis")
    assert list(cli.discover_inputs(tmp_path)) == ["Finland"]


def test_read_rows_preserves_blanks_and_every_column(cli, tmp_path):
    path = _write_input(tmp_path, "finland", rows=2)
    columns, rows = cli.read_rows(path)
    assert columns == HEADER
    assert len(rows) == 2
    assert rows[0]["new_entity"] == "", "a blank field must read back blank"
    assert rows[0]["researcher_new_persons"] == "[]"


def test_dry_run_reports_counts_without_writing_anything(cli, tmp_path, capsys):
    _write_input(tmp_path, "finland", rows=3)
    rc = cli.main(["--private-root", str(tmp_path), "--country", "Finland", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"Finland": 3' in out
    assert not (ROOT / "outputs" / "ep24_bootstrap_manifest.json").exists()


def test_limit_caps_rows_per_country(cli, tmp_path, capsys):
    _write_input(tmp_path, "finland", rows=10)
    rc = cli.main(["--private-root", str(tmp_path), "--country", "Finland",
                   "--limit", "4", "--dry-run"])
    assert rc == 0
    assert '"Finland": 4' in capsys.readouterr().out


def test_country_filter_selects_only_that_country(cli, tmp_path, capsys):
    _write_input(tmp_path, "finland")
    _write_input(tmp_path, "poland")
    rc = cli.main(["--private-root", str(tmp_path), "--country", "Poland", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Poland" in out and "Finland" not in out


def test_cli_never_writes_the_private_source_file(cli, tmp_path):
    path = _write_input(tmp_path, "finland", rows=2)
    before = path.read_bytes()
    cli.main(["--private-root", str(tmp_path), "--country", "Finland", "--dry-run"])
    assert path.read_bytes() == before, "the authoritative private input must never change"


def test_bootstrap_merges_run_before_any_stage(cli, tmp_path):
    """The canonical fields exist on the bootstrapped rows the CLI produces."""
    import ep24_bootstrap as bootstrap

    _write_input(tmp_path, "finland", rows=2)
    columns, rows = cli.read_rows(
        tmp_path / "analysis" / "ep24_reprocess" / "data" / "to_reprocess" / "ep24_finland.csv"
    )
    bootstrapped = bootstrap.bootstrap_rows(rows, country="Finland")
    for row in bootstrapped:
        assert "entities" in row and "themes" in row
        assert row["themes"] == "synthetic theme"
        assert row["entities"] == "", "the synthetic rows carry no entity labels"
        assert row["new_entity"] == "" and row["researcher_new_persons"] == "[]"
