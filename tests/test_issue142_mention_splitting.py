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


def test_split_mentions_matches_the_authoritative_separator_set():
    """A comma-separated cell must yield one mention per actor, as enrich/RDF do."""
    from roihu_enrich import split_values
    from roihu_rdf import split_list

    cell = "Sanna Marin, Petteri Orpo"
    assert E._split_mentions(cell) == split_list(cell) == split_values(cell)
    assert E._split_mentions(cell) == ["Sanna Marin", "Petteri Orpo"]


def test_comma_separated_cell_resolves_every_actor_not_one_mention():
    """The regression: the whole cell used to become one UNRESOLVED mention."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "new_entity": "Sanna Marin, Petteri Orpo"}]
    )
    summary = E.resolve_dataframe(
        frame,
        fi_registry(),
        country="FI",
        language="fi",
        mention_columns=("new_entity",),
    )

    assert summary["total"] == 2, f"expected 2 mentions, got {summary['total']}"
    assert summary["decisions"].get("RESOLVED") == 2
    assert "UNRESOLVED" not in summary["decisions"]

    ids = json.loads(frame.iloc[0]["ep24_entity_ids"])
    assert ids == ["CB-marin", "CB-orpo"], ids
    # the original wording is untouched -- the issue's central requirement
    assert frame.iloc[0]["new_entity"] == "Sanna Marin, Petteri Orpo"


def test_pipe_and_semicolon_cells_keep_working():
    """The new comma handling must not break the separators that already worked."""
    assert E._split_mentions("Orpo; Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("Orpo|Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("A\nB") == ["A", "B"]
    assert E._split_mentions("Orpo; Orpo") == ["Orpo"]


def test_acronym_containing_commas_is_not_an_actor_cell():
    """A lone token must survive intact; splitting is for list cells only."""
    assert E._split_mentions("Petteri Orpo") == ["Petteri Orpo"]
    assert E._split_mentions("[]") == []
    assert E._split_mentions("") == []


def test_entities_column_commas_resolve_every_actor():
    """The real column name from the corpus, not just the synthetic alias."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"country": "FI", "entities": "Sanna Marin, Petteri Orpo"}]
    )
    summary = E.resolve_dataframe(
        frame, fi_registry(), country="FI", language="fi", mention_columns=("entities",)
    )
    assert summary["decisions"].get("RESOLVED") == 2
    assert json.loads(frame.iloc[0]["ep24_entity_ids"]) == ["CB-marin", "CB-orpo"]
