"""Tests for the EP24 codebook coverage auditor (issue #72, scripts/ep24/codebook_coverage.py).

The auditor exists so every agent reviewing a country produces COMPARABLE
numbers instead of re-inventing the analysis. These tests pin the two properties
that matter:

1. the measurements are correct on a synthetic fixture, and
2. the auditor is READ-ONLY and never resolves ambiguity.

Fixtures are synthetic: no private codebook content, rows, notes or personal data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "codebook_coverage", ROOT / "scripts" / "ep24" / "codebook_coverage.py"
)
assert _SPEC and _SPEC.loader
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


def _write(root: Path, name: str, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(
            {"schema": "test-fixture", "country_code": "HR", "entries": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "codebooks"
    _write(
        root,
        "ep24_hr_private.json",
        [
            # one actor fragmented across entries, no aliases (the D1 shape)
            {"kind": "entity", "label": "Andrej Plenković", "aliases": [], "metadata": {"observations": 64}},
            {"kind": "entity", "label": "Plenković", "aliases": [], "metadata": {"observations": 26}},
            {"kind": "entity", "label": "Andrija Plenković", "aliases": [], "metadata": {"observations": 3}},
            # a party WITH aliases (the healthy shape)
            {
                "kind": "entity",
                "label": "Croatian Democratic Union",
                "aliases": ["HDZ", "Hrvatska demokratska zajednica"],
                "metadata": {"observations": 10},
            },
            # near-duplicate themes (the D6 shape)
            {"kind": "theme", "label": "Croatian identity"},
            {"kind": "theme", "label": "Croatian National Identity"},
            # an unrelated theme that must NOT be grouped
            {"kind": "theme", "label": "Climate and energy"},
        ],
    )
    return root


def test_alias_coverage_counts_entries_without_aliases(tmp_path: Path) -> None:
    report = coverage.audit(_fixture(tmp_path), "HR")
    layer = report["layers"]["ep24_hr_private.json"]
    cov = layer["alias_coverage"]
    assert cov["entries"] == 7
    # exactly one fixture entry carries aliases, so six do not
    assert cov["entries_without_aliases"] == 6, "6 of 7 entries carry no aliases"
    assert cov["aliases_total"] == 2


def test_entity_fragmentation_groups_same_surname_and_sums_observations(tmp_path: Path) -> None:
    report = coverage.audit(_fixture(tmp_path), "HR")
    frag = report["layers"]["ep24_hr_private.json"]["entity_fragmentation"]
    assert frag, "the fixture must contain a fragmented surname"
    top = frag[0]
    assert top["key"] == "plenkovic"
    assert top["count"] == 3, "three entries share the trailing name Plenković"
    assert top["observations"] == 93, "64 + 26 + 3"


def test_kind_breakdown_makes_missing_party_entities_visible(tmp_path: Path) -> None:
    report = coverage.audit(_fixture(tmp_path), "HR")
    kinds = report["layers"]["ep24_hr_private.json"]["kind_breakdown"]
    assert kinds == {"entity": 4, "theme": 3}


def test_theme_near_duplicates_are_flagged_but_distinct_themes_are_not(tmp_path: Path) -> None:
    report = coverage.audit(_fixture(tmp_path), "HR")
    dups = report["layers"]["ep24_hr_private.json"]["theme_near_duplicates"]
    pairs = {frozenset((d["a"], d["b"])) for d in dups}
    assert frozenset(("Croatian identity", "Croatian National Identity")) in pairs
    # A distinct theme must not be grouped with the identity themes.
    for d in dups:
        assert "Climate and energy" not in (d["a"], d["b"])


def test_audit_is_read_only(tmp_path: Path) -> None:
    """The auditor must not modify the codebook it reads."""
    root = _fixture(tmp_path)
    path = root / "ep24_hr_private.json"
    before = path.read_bytes()
    coverage.audit(root, "HR")
    assert path.read_bytes() == before, "the auditor modified the codebook"


def test_missing_country_is_a_clean_error(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        coverage.audit(root, "ZZ")


def test_canonicalisation_ignores_accents_and_case_for_grouping(tmp_path: Path) -> None:
    """Grouping folds accents so `Plenkovic` and `Plenković` group together."""
    assert coverage._fold("Plenković") == coverage._fold("Plenkovic")
    assert coverage._fold("Croatian  Identity!") == coverage._fold("croatian identity")
