"""Regression guard: Step 6 abstention must not cost the RDF its coding (#180).

``THEORY`` makes abstention a first-class outcome: when the source does not
evidence an affect, Step 6 must leave it unspecified rather than fabricate an
emotion from polarity. The legacy ``formula_of_populism_us`` /
``formula_of_populism_frontier`` columns are the RDF compatibility surface, so
the element has to survive there even when there is no affect.

Two ways to get this wrong, and both have been in the tree:

* ``f"{element}^{affect}"`` with no affect fabricates an emotion;
* omitting the line entirely loses the element, so the coding silently vanishes
  from the graph and abstention costs the analysis its finding. That was #180.

The correct form is a bare ``element`` line: it carries the evidenced element and
asserts no affect. Both RDF consumers must agree on that reading.

Synthetic fixtures only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import roihu_csv_rdf  # noqa: E402
import roihu_rdf  # noqa: E402
from roihu_populism import (  # noqa: E402
    AffectObservation,
    EP24DiscourseAnalysis,
    FrontierConstruct,
    UsConstruct,
    compatibility_columns,
)

BASE = "https://example.org"


def _coding_warnings(warnings):
    """Ignore the unrelated row-identity advisory; we only assert about coding."""
    return [w for w in warnings if "formula_of_populism" in w or "malformed" in w.lower()]



def _result(affects):
    return EP24DiscourseAnalysis(
        analysis_md="",
        us_constructs=[
            UsConstruct(label="the people", text_span="we the people", confidence=0.8)
        ],
        frontier_constructs=[
            FrontierConstruct(
                label="elites",
                them_side="elites",
                relation="antagonistic_frontier",
                text_span="them",
                confidence=0.7,
            )
        ],
        affects=affects,
    )


# --------------------------------------------------------------------------- #
# Step 6 projection
# --------------------------------------------------------------------------- #

def test_abstention_keeps_the_element_and_invents_no_affect():
    us, frontier = compatibility_columns(_result([]))
    assert us == "the people", "an evidenced Us must not be dropped for lack of an affect"
    assert frontier == "elites"
    assert "^" not in us and "^" not in frontier, "no affect may be fabricated"


def test_evidenced_affect_keeps_the_historical_form():
    us, frontier = compatibility_columns(
        _result(
            [
                AffectObservation(
                    label="hope-for-the-people",
                    affect="hope",
                    target="the people",
                    text_span="we",
                    confidence=0.9,
                )
            ]
        )
    )
    assert us == "the people^hope"
    assert frontier == "elites"


# --------------------------------------------------------------------------- #
# CSV RDF projection (the module whose gate rejected the bare line)
# --------------------------------------------------------------------------- #

def test_csv_rdf_represents_a_bare_element_as_a_coding():
    """The regression: this used to warn and drop, leaving element triples at 0."""
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": "the people", "formula_of_populism_frontier": "elites"},
        base=BASE,
        project="p",
        dataset="d",
        row_number=1,
    )
    assert _coding_warnings(warnings) == [], f"a bare element is a valid coding: {warnings}"
    assert graph.count("LaclauCoding") == 2
    assert "the people" in graph
    assert "elites" in graph
    # an abstention is the ABSENCE of an affect claim, not an empty literal
    assert "/affect>" not in graph


def test_csv_rdf_unchanged_for_evidenced_affect():
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": "the people^hope", "formula_of_populism_frontier": "elites"},
        base=BASE,
        project="p",
        dataset="d",
        row_number=1,
    )
    assert _coding_warnings(warnings) == []
    assert graph.count("LaclauCoding") == 2
    assert "hope" in graph


def test_csv_rdf_still_rejects_a_genuinely_ambiguous_line():
    """Two separators cannot be read unambiguously, and must still warn."""
    _doc, _record, _graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": "a^b^c"}, base=BASE, project="p", dataset="d", row_number=1
    )
    assert len(_coding_warnings(warnings)) == 1


def test_csv_rdf_still_rejects_an_empty_element():
    _doc, _record, _graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": "^anger"}, base=BASE, project="p", dataset="d", row_number=1
    )
    assert len(_coding_warnings(warnings)) == 1


# --------------------------------------------------------------------------- #
# both consumers agree on the bare line
# --------------------------------------------------------------------------- #

def test_both_rdf_consumers_read_a_bare_element_the_same_way():
    """roihu_rdf already documented this reading; the CSV projection must match."""
    pairs = roihu_rdf.parse_populism_elements("the people")
    assert pairs == [("the people", "")], (
        "the graph exporter already treats a separatorless line as element-with-no-affect"
    )
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": "the people"}, base=BASE, project="p", dataset="d", row_number=1
    )
    assert _coding_warnings(warnings) == []
    assert "the people" in graph


def test_end_to_end_abstention_reaches_the_csv_rdf_graph():
    """The issue's own reproduction: abstention used to yield element triples = 0."""
    us, frontier = compatibility_columns(_result([]))
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": us, "formula_of_populism_frontier": frontier},
        base=BASE,
        project="p",
        dataset="d",
        row_number=1,
    )
    assert _coding_warnings(warnings) == []
    assert graph.count("LaclauCoding") == 2, "abstention must not cost the coding"


@pytest.mark.parametrize("column", ["formula_of_populism_us", "formula_of_populism_frontier"])
def test_blank_cells_still_produce_nothing(column):
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {column: ""}, base=BASE, project="p", dataset="d", row_number=1
    )
    assert _coding_warnings(warnings) == []
    assert graph.count("LaclauCoding") == 0
