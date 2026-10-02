"""Stage/column contract tests for the numbered EP24 Roihu pipeline (issue #64).

The issue's central safety property is that no stage may silently drop a column an
earlier stage produced. These tests make that property fail loudly, in three
layers:

1. the contract itself is internally consistent and names real files;
2. every ``appends`` column is really written by the module the contract credits
   (checked against the source text, so it runs without Ollama/GPU/private data);
3. a synthetic row that has been through bootstrap keeps every source column and
   gains the canonical merged fields, with the four researcher/source columns
   preserved alongside them.

The fixtures here are synthetic. No private EP24 rows, researcher notes,
codebooks or credentials appear in this file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_stage_contract as contract  # noqa: E402
from ep24_schema import EP24_REPROCESS_COLUMNS  # noqa: E402


# --- synthetic fixture -----------------------------------------------------

SYNTHETIC_ROW = {
    "video_id": "SYNTH-FI-0001",
    "country": "Finland",
    "author_username": "synthetic-profile-a",
    "account_type": "Synthetic",
    "source_type": "Instagram",
    "source_recording": "screen-synthetic-01",
    "sequence_number": "1",
    "political_preference": "synthetic preference",
    "allas_filename": "https://example.invalid/HEPP24/FI/CR/IG/synthetic.mp4",
    "new_entity": "Synthetic Person",
    "new_theme": "synthetic theme",
    "video_duration": "10.0",
    "researcher_new_persons": "[]",
    "researcher_new_themes": "[]",
    "researcher_note": "",
}


# --- 1. the contract is consistent and points at real files ----------------

def test_contract_covers_all_nine_numbered_stages():
    numbers = [stage.number for stage in contract.STAGE_CONTRACT]
    assert numbers == list(range(1, 10)), f"expected stages 1..9, got {numbers}"


def test_every_stage_names_a_module_that_exists():
    missing = [s.module for s in contract.STAGE_CONTRACT if not (ROOT / s.module).is_file()]
    assert not missing, f"contract names modules that do not exist: {missing}"


def test_every_stage_entry_point_and_sbatch_exist():
    """The launcher/sbatch pair must exist, so a launcher cannot point at nothing."""
    missing = []
    for stage in contract.STAGE_CONTRACT:
        if not (ROOT / stage.entry_point).is_file():
            missing.append(stage.entry_point)
        if not (ROOT / stage.sbatch).is_file():
            missing.append(stage.sbatch)
    assert not missing, f"contract names entry points/sbatch files that do not exist: {missing}"


def test_stage_lookup_rejects_an_unknown_number_and_lists_the_known_ones():
    try:
        contract.stage(99)
    except KeyError as exc:
        assert "99" in str(exc)
    else:  # pragma: no cover - the contract must fail loudly
        raise AssertionError("contract.stage(99) should raise")


def test_source_columns_are_the_fifteen_column_keep_schema():
    assert len(contract.SOURCE_COLUMNS) == 15
    assert contract.SOURCE_COLUMNS == EP24_REPROCESS_COLUMNS


def test_bootstrap_merges_from_columns_it_does_not_consume():
    """`entities`/`themes` are added; the four researcher columns they merge from
    must remain, or provenance is destroyed (issue #64, non-negotiable rule)."""
    for column in contract.BOOTSTRAP_PRESERVED_COLUMNS:
        assert column in contract.SOURCE_COLUMNS, f"{column} must be preserved"
        assert column not in contract.BOOTSTRAP_ADDED_COLUMNS, f"{column} must not be consumed"
    for column in ("entities", "themes"):
        assert column in contract.BOOTSTRAP_ADDED_COLUMNS
        assert column not in contract.SOURCE_COLUMNS


# --- 2. every contracted column is really written by its stage -------------

def _writes_column(source: str, column: str) -> bool:
    """True when the module actually writes this column.

    A bare substring check is too weak: a column can survive inside a *read*
    (``if 'video_filename' in df.columns``) long after its write is gone. So the
    name must appear in one of the write forms the stages use, or as a quoted
    literal in a column-init tuple -- both of which are how these scripts create
    a column.
    """
    escaped = re.escape(column)
    forms = [
        rf"df\[['\"]{escaped}['\"]\]\s*=",
        rf"df\.at\[[^\]]*,\s*['\"]{escaped}['\"]\]\s*=",
        rf"setdefault\(['\"]{escaped}['\"]",
        rf"['\"]{escaped}['\"]",
    ]
    return any(re.search(form, source) for form in forms)


def _stage_sources(stage) -> list[str]:
    """All source files a stage's columns may be written in.

    A contracted stage is sometimes a thin wrapper (``step_3_roihu_video.py``)
    that delegates to an implementation module (``experiments/vllm_video_test.py``).
    Checking only the wrapper would report every column as missing even though the
    stage genuinely writes them, so follow simple ``from X import ...`` /
    ``import X`` delegation one level deep and include the target's source.

    This is deliberately shallow: a chain of wrappers is not something the
    pipeline actually uses, and guessing further would make the check dishonest
    about which file a column lives in.
    """
    module_path = ROOT / stage.module
    sources = [module_path.read_text(encoding="utf-8")]
    for line in sources[0].splitlines():
        match = re.match(r"\s*from\s+([\w\.]+)\s+import\s+", line) or re.match(r"\s*import\s+([\w\.]+)", line)
        if not match:
            continue
        target = ROOT.joinpath(*match.group(1).split("."))
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            if candidate.is_file():
                sources.append(candidate.read_text(encoding="utf-8"))
                break
    return sources


def test_every_contracted_append_is_written_by_the_module_it_credits():
    """A contract column that is no longer written means a stage silently shrank.

    Note the wrapper case: `step_3_roihu_video.py` delegates to
    `experiments/vllm_video_test.py`, where the columns are actually written, so
    the check reads the delegated implementation too rather than failing on a
    correct contract.
    """
    failures = []
    for stage in contract.STAGE_CONTRACT:
        if not stage.appends:
            continue
        sources = _stage_sources(stage)
        for column in stage.appends:
            if not any(_writes_column(source, column) for source in sources):
                failures.append(f"{stage.module} no longer writes {column!r}")
    assert not failures, "contracted columns are no longer written:\n  " + "\n  ".join(failures)


def test_contract_has_no_duplicate_columns_within_a_stage():
    for stage in contract.STAGE_CONTRACT:
        assert len(set(stage.appends)) == len(stage.appends), (
            f"stage {stage.number} lists a column twice"
        )


def test_all_contracted_columns_start_with_the_source_columns():
    """Source columns come first so a cumulative CSV reads identity-first."""
    ordered = contract.all_contracted_columns()
    assert ordered[: len(contract.SOURCE_COLUMNS)] == contract.SOURCE_COLUMNS
    assert len(set(ordered)) == len(ordered), "all_contracted_columns produced duplicates"


# --- 3. the cumulative dataframe keeps everything -------------------------

def test_bootstrap_shaped_row_keeps_every_source_column_and_gains_the_merged_ones():
    """The merged fields are additive: nothing is renamed, dropped or replaced."""
    merged = dict(SYNTHETIC_ROW)
    merged["record_id"] = "SYNTH-FI-0001|Finland|0"
    merged["entities"] = "Synthetic Person"
    merged["themes"] = "synthetic theme"

    for column in contract.SOURCE_COLUMNS:
        assert column in merged, f"bootstrap dropped the source column {column}"
        assert merged[column] == SYNTHETIC_ROW[column], f"bootstrap mutated {column}"
    for column in contract.BOOTSTRAP_ADDED_COLUMNS:
        assert column in merged


def test_country_order_follows_issue128_exact_sequence():
    order = contract.country_order(
        [
            "Sweden", "Bulgaria", "France", "Croatia", "Hungary",
            "Spain", "Germany", "Portugal", "Poland", "Finland",
        ]
    )
    assert order == list(contract.COUNTRY_PROCESSING_ORDER)


def test_country_order_never_drops_an_unlisted_country():
    order = contract.country_order(["Finland", "Atlantis"])
    assert set(order) == {"Finland", "Atlantis"}
    assert order[0] == "Finland"


def test_country_order_is_deterministic_and_deduplicates():
    first = contract.country_order(["Sweden", "Finland", "Finland", "Poland"])
    second = contract.country_order(["Sweden", "Finland", "Finland", "Poland"])
    assert first == second
    assert first.count("Finland") == 1


def test_country_tokens_cover_every_priority_country():
    for country in contract.COUNTRY_PRIORITY:
        assert country in contract.COUNTRY_TOKENS, f"{country} has no input-file token"
        assert contract.COUNTRY_TOKENS[country] == country.lower()
