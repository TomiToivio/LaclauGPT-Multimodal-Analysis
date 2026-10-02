"""Regression guards for canonical mention splitting (#142 / #21).

The #21 migration made the input annotation cells **JSON lists** written by
``analysis/ep24_reprocess/scripts/canonicalize_input_schema.py`` (in the private
repo), so the authoritative splitter parses JSON first and only then falls back
to plain delimiters. These tests pin that behaviour, plus two defects found along
the way:

1. ``_split_mentions`` once split only on newlines and semicolons while
   ``roihu_enrich.split_values`` / ``roihu_rdf.split_list`` used a comma,
   because the postprocess stage wrote ``', '.join(...)``. Then a cell such as
   ``"Sanna Marin, Petteri Orpo"`` was resolved as ONE mention and both actors
   were lost to the unresolved queue. That is no longer reachable for canonical
   input because the cell is now a JSON list, and a comma must NOT be an
   implicit delimiter anyway -- the canonical theme list contains eight names
   that carry a comma as part of the name (``war, conflict and military``,
   ``populism, peopleism``, ...).

2. The JSON-first rewrite wrote the newline escapes doubled
   (``text.replace("\\\\r", "\\\\n").split("\\\\n")``), which matches the two
   character sequences backslash-r / backslash-n rather than an actual newline.
   A cell containing a real newline therefore stopped splitting. Pinned below.

Synthetic fixtures only: no private research data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ep24_entities as E  # noqa: E402


def fi_registry() -> E.EntityRegistry:
    registry = E.EntityRegistry()
    registry.add(
        E.EntityRecord(
            entity_id="CB-marin",
            canonical_name="Sanna Marin",
            aliases=["Marin"],
            country="FI",
            entity_type="person",
        )
    )
    registry.add(
        E.EntityRecord(
            entity_id="CB-orpo",
            canonical_name="Petteri Orpo",
            aliases=["Orpo", "Pääministeri Orpo"],
            country="FI",
            entity_type="person",
        )
    )
    return registry


# --------------------------------------------------------------------------- #
# canonical (JSON list) cells
# --------------------------------------------------------------------------- #

def test_json_list_cell_yields_every_annotation():
    """The canonical input format: one JSON list per cell."""
    assert E._split_mentions('["Sanna Marin", "Petteri Orpo"]') == [
        "Sanna Marin",
        "Petteri Orpo",
    ]


def test_comma_inside_a_label_is_never_a_delimiter():
    """Eight canonical themes contain a comma as part of the name."""
    cell = json.dumps(["war, conflict and military", "populism, peopleism"])
    assert E._split_mentions(cell) == ["war, conflict and military", "populism, peopleism"]
    # ...and in the plain-text fallback a comma is still not a delimiter.
    assert E._split_mentions("war, conflict and military") == ["war, conflict and military"]


def test_json_cell_resolves_every_actor():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "entities": json.dumps(["Sanna Marin", "Petteri Orpo"])}]
    )
    summary = E.resolve_dataframe(
        frame, fi_registry(), country="FI", language="fi", mention_columns=("entities",)
    )
    assert summary["total"] == 2, f"expected 2 mentions, got {summary['total']}"
    assert summary["decisions"].get("RESOLVED") == 2
    assert json.loads(frame.iloc[0]["ep24_entity_ids"]) == ["CB-marin", "CB-orpo"]
    # the original wording is untouched -- the issue's central requirement
    assert frame.iloc[0]["entities"] == json.dumps(["Sanna Marin", "Petteri Orpo"])


def test_json_duplicates_collapse_and_blank_members_drop():
    assert E._split_mentions('["Orpo", "Orpo"]') == ["Orpo"]
    assert E._split_mentions('["Orpo", ""]') == ["Orpo"]
    assert E._split_mentions("[]") == []


# --------------------------------------------------------------------------- #
# plain-text fallback (older hand-authored fixtures)
# --------------------------------------------------------------------------- #

def test_plain_delimiters_still_work():
    assert E._split_mentions("Orpo; Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("Orpo|Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("Orpo; Orpo") == ["Orpo"]
    assert E._split_mentions("Petteri Orpo") == ["Petteri Orpo"]
    assert E._split_mentions("") == []


def test_real_newlines_split_the_plain_text_fallback():
    """Regression: the escapes were doubled, so a real newline stopped splitting."""
    assert E._split_mentions("A\nB") == ["A", "B"]
    assert E._split_mentions("A\r\nB") == ["A", "B"]
    assert E._split_mentions("A\nB\nC") == ["A", "B", "C"]


def test_entities_column_json_resolves_every_actor():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "entities": '["Sanna Marin", "Petteri Orpo"]'}]
    )
    summary = E.resolve_dataframe(
        frame, fi_registry(), country="FI", language="fi", mention_columns=("entities",)
    )
    assert summary["decisions"].get("RESOLVED") == 2
    assert json.loads(frame.iloc[0]["ep24_entity_ids"]) == ["CB-marin", "CB-orpo"]
