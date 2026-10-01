"""Regression tests for the codebook coverage auditor's entity-fragmentation key.

Found while auditing Finland for issue #72, a country the original tests in
``test_ep24_codebook_coverage_tool.py`` did not exercise.

The defect
----------
``entity_fragmentation`` keyed on the **last token longer than 3 characters**.
For English-labelled entries that token is frequently a *kind* word rather than a
name, so unrelated organisations collapsed into one group. Finland's largest
reported group was 25 unrelated entries under the single key ``party``:

    Brothers of Italy party / Centre Party / Finnish Social Democratic Party / ...

reported as ``obs=209  n=25``. That is not a fragmentation group, it is an
artifact -- and because it outranked everything, it **masked the real signal**.
After skipping kind words, Finland reports 36 genuine groups, led by
Perussuomalaiset (5 surface forms) and Kokoomus (4).

Blast radius: PT, FI, PL, DE, SE were affected; HR, ES, BG, FR, HU were immune
(none of their largest groups keyed on a kind word). The original tests were
verified against Croatia -- one of the immune countries -- which is why this
survived.

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


def _book(root: Path, name: str, *, country_code: str, country: str, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": country_code,
                "country": country,
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, entries: list[dict]) -> Path:
    root = tmp_path / "codebooks"
    _book(root, "ep24_xx_private.json", country_code="XX", country="Example", entries=entries)
    return root


def _frag(root: Path) -> list[dict]:
    return coverage.audit(root, "XX")["layers"]["ep24_xx_private.json"]["entity_fragmentation"]


# ------------------------------------------------------- the kind-word false merge


def test_party_labels_do_not_all_collapse_under_the_kind_word(tmp_path: Path) -> None:
    """Three distinct parties must not become one fragmentation group.

    Before the fix every label ending in the literal word `party` produced the
    surname key `party`, so this fixture reported a single 3-entry group at
    obs=14 -- the Finland shape in miniature.
    """
    root = _fixture(
        tmp_path,
        [
            {"kind": "entity", "label": "Centre Party", "metadata": {"observations": 5}},
            {"kind": "entity", "label": "Finns Party", "metadata": {"observations": 7}},
            {"kind": "entity", "label": "Brothers of Italy party", "metadata": {"observations": 2}},
        ],
    )
    frag = _frag(root)
    keys = {row["key"] for row in frag}
    assert "party" not in keys, "a kind word must never be a grouping key"
    # Each label ends in a different real name, so no group should form at all.
    assert frag == [], f"distinct parties were falsely merged: {frag}"


def test_real_fragmentation_is_still_detected_beneath_the_kind_word(tmp_path: Path) -> None:
    """The fix must expose the real signal, not merely suppress the false one.

    `Perussuomalaiset` and `Perussuomalaiset party` are the same actor twice, so
    they must group; `Centre Party` must not join them.
    """
    root = _fixture(
        tmp_path,
        [
            {"kind": "entity", "label": "Perussuomalaiset", "metadata": {"observations": 30}},
            {"kind": "entity", "label": "Perussuomalaiset party", "metadata": {"observations": 12}},
            {"kind": "entity", "label": "Centre Party", "metadata": {"observations": 5}},
        ],
    )
    by_key = {row["key"]: row for row in _frag(root)}
    assert "perussuomalaiset" in by_key, "the same actor under two labels must still group"
    assert by_key["perussuomalaiset"]["count"] == 2
    assert by_key["perussuomalaiset"]["observations"] == 42
    assert not any("centre" in k for k in by_key), "an unrelated party must not be merged in"


def test_a_genuine_surname_key_is_unaffected_by_the_role_word_skip(tmp_path: Path) -> None:
    """Skipping kind words must not change ordinary surname grouping."""
    root = _fixture(
        tmp_path,
        [
            {"kind": "entity", "label": "Andrej Plenković", "metadata": {"observations": 64}},
            {"kind": "entity", "label": "Plenković", "metadata": {"observations": 26}},
        ],
    )
    frag = _frag(root)
    assert frag and frag[0]["key"] == "plenkovic"
    assert frag[0]["observations"] == 90


def test_role_words_do_not_hide_a_trailing_surname(tmp_path: Path) -> None:
    """`<Name> Party` keeps the name as the key, not the kind word."""
    root = _fixture(
        tmp_path,
        [
            {"kind": "entity", "label": "Vihreät Party", "metadata": {"observations": 4}},
            {"kind": "entity", "label": "Vihreät", "metadata": {"observations": 6}},
        ],
    )
    frag = _frag(root)
    assert frag and frag[0]["key"] == "vihreat"


@pytest.mark.parametrize(
    "label",
    ["Party", "Coalition", "Movement", "The Party", "An Alliance"],
)
def test_a_label_that_is_only_kind_words_keys_to_nothing(label: str) -> None:
    """Such a label has no name to group on; it must return "" rather than crash."""
    assert coverage._surname_key(label) == ""


def test_empty_and_blank_labels_are_safe() -> None:
    assert coverage._surname_key("") == ""
    assert coverage._surname_key("   ") == ""


def test_kind_words_are_not_counted_as_entities_for_fragmentation(tmp_path: Path) -> None:
    """A book of only `X Party` labels must produce no fragmentation groups.

    This is the end-to-end shape of the Finland defect: before the fix this
    fixture produced one group keyed `party`.
    """
    root = _fixture(
        tmp_path,
        [
            {"kind": "entity", "label": "Centre Party", "metadata": {"observations": 1}},
            {"kind": "entity", "label": "Green Party", "metadata": {"observations": 2}},
            {"kind": "entity", "label": "Left Party", "metadata": {"observations": 3}},
        ],
    )
    assert _frag(root) == []


def test_fix_is_read_only(tmp_path: Path) -> None:
    """The auditor must not modify the codebook it reads."""
    root = _fixture(
        tmp_path,
        [{"kind": "entity", "label": "Centre Party", "metadata": {"observations": 5}}],
    )
    path = root / "ep24_xx_private.json"
    before = path.read_bytes()
    coverage.audit(root, "XX")
    assert path.read_bytes() == before
