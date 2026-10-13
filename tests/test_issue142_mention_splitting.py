"""Regression guard: mention splitting must match the other layers (#142).

``ep24_entities._split_mentions`` split only on newlines and semicolons, but the
authoritative splitter -- ``roihu_enrich.split_values`` and
``roihu_rdf.split_list`` -- treats a comma as a separator, because the
postprocess stage writes the cell with ``', '.join(...)``.

Consequence on real data: a cell such as ``"Sanna Marin, Petteri Orpo"`` was
resolved as a **single** mention, so it matched nothing and *both* actors were
lost to the unresolved queue -- precisely the fragmentation the layer exists to
prevent. The downstream RDF projection then has no stable IDs to key on and
falls back to surface-string nodes, which is the duplicate-node outcome #142's
acceptance criterion forbids.

Measured on the private corpus: 2,375 of the Finland ``entities`` cells and
3,496 of the Hungary cells carry commas.

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


def test_split_mentions_matches_the_authoritative_json_contract():
    from roihu_enrich import split_values
    from roihu_rdf import split_list

    cell = '["Sanna Marin", "Petteri Orpo"]'
    assert E._split_mentions(cell) == split_list(cell) == split_values(cell)
    assert E._split_mentions(cell) == ["Sanna Marin", "Petteri Orpo"]


def test_json_canonical_cell_resolves_every_actor():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "entities": '["Sanna Marin", "Petteri Orpo"]'}]
    )
    summary = E.resolve_dataframe(frame, fi_registry(), country="FI", language="fi")

    assert summary["total"] == 2
    assert summary["decisions"].get("RESOLVED") == 2
    assert "UNRESOLVED" not in summary["decisions"]

    ids = json.loads(frame.iloc[0]["ep24_entity_ids"])
    assert ids == ["CB-marin", "CB-orpo"], ids
    assert frame.iloc[0]["entities"] == '["Sanna Marin", "Petteri Orpo"]'


def test_pipe_and_semicolon_cells_keep_working():
    """The new comma handling must not break the separators that already worked."""
    assert E._split_mentions("Orpo; Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("Orpo|Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("A\nB") == ["A", "B"]
    assert E._split_mentions("Orpo; Orpo") == ["Orpo"]


def test_comma_inside_one_json_label_is_preserved():
    assert E._split_mentions('["Example Coalition, National Wing"]') == [
        "Example Coalition, National Wing"
    ]
    assert E._split_mentions("Petteri Orpo") == ["Petteri Orpo"]
    assert E._split_mentions("[]") == []
    assert E._split_mentions("") == []


def test_entities_column_json_resolves_every_actor():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "entities": '["Sanna Marin", "Petteri Orpo"]'}]
    )
    summary = E.resolve_dataframe(frame, fi_registry(), country="FI", language="fi")
    assert summary["decisions"].get("RESOLVED") == 2
    assert json.loads(frame.iloc[0]["ep24_entity_ids"]) == ["CB-marin", "CB-orpo"]
