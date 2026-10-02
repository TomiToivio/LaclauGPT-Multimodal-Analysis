"""Regression guard: Step 6 must emit the bare element so abstention survives (#180).

#180's failure is end-to-end: Step 6 abstains (no evidenced affect), and by the
time the RDF projection runs there is nothing left to project, so
``element triples = 0``. The RDF projection half of that was fixed separately
(#181) -- it now accepts a bare element as a coding with no affect claim. This
file guards the half that was still missing: Step 6 has to *emit* the element.

Both wrong answers have been in the tree and neither is acceptable:

* ``f"{label}^{affect}"`` with no affect fabricates an emotion;
* dropping the line loses the element, so abstention costs the analysis its
  finding -- which quietly pressures the model back toward fabricating affect.

A bare ``element`` line carries the evidenced element and asserts no affect.

Synthetic fixtures only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import roihu_csv_rdf  # noqa: E402
from roihu_populism import (  # noqa: E402
    AffectObservation,
    EP24DiscourseAnalysis,
    FrontierConstruct,
    UsConstruct,
    compatibility_columns,
)

BASE = "https://example.org"


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


def _coding_warnings(warnings):
    """Isolate coding warnings; the row-identity advisory is unrelated here."""
    return [w for w in warnings if "formula_of_populism" in w]


# --------------------------------------------------------------------------- #
# Step 6 output
# --------------------------------------------------------------------------- #

def test_abstention_emits_the_bare_element_and_invents_no_affect():
    us, frontier = compatibility_columns(_result([]))
    assert us == "the people", "an evidenced Us must not be dropped for lack of an affect"
    assert frontier == "elites"
    assert "^" not in us and "^" not in frontier, "no affect may be fabricated"


def test_evidenced_affect_keeps_the_historical_form():
    us, _frontier = compatibility_columns(
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


def test_an_ambiguous_affect_is_omitted_rather_than_emitted():
    """An affect with '^' or a newline would parse differently in the consumers."""
    for bad in ("a^b", "a\nb"):
        us, _frontier = compatibility_columns(
            _result(
                [
                    AffectObservation(
                        label="x", affect=bad, target="the people",
                        text_span="we", confidence=0.9,
                    )
                ]
            )
        )
        assert us == "the people", f"affect {bad!r} must not be emitted into the line"
        assert us.count("^") == 0


def test_elements_with_no_label_are_skipped():
    us, frontier = compatibility_columns(_result([]))
    assert us.splitlines() == ["the people"]
    assert frontier.splitlines() == ["elites"]


# --------------------------------------------------------------------------- #
# the issue's end-to-end reproduction
# --------------------------------------------------------------------------- #

def test_issue_180_reproduction_abstention_reaches_the_rdf_graph():
    """The defect: this produced element triples = 0 before the fix."""
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
    assert graph.count("/element>") == 2, "the element must reach the graph"
    assert "the people" in graph
    assert "elites" in graph
    # ...and still no invented emotion.
    assert "/affect>" not in graph


def test_with_affect_the_graph_gains_the_affect_triple():
    us, frontier = compatibility_columns(
        _result(
            [
                AffectObservation(
                    label="hope-for-the-people", affect="hope", target="the people",
                    text_span="we", confidence=0.9,
                )
            ]
        )
    )
    _doc, _record, graph, warnings = roihu_csv_rdf.project_row(
        {"formula_of_populism_us": us, "formula_of_populism_frontier": frontier},
        base=BASE, project="p", dataset="d", row_number=1,
    )
    assert _coding_warnings(warnings) == []
    assert graph.count("/element>") == 2
    assert graph.count("/affect>") == 1
    assert "hope" in graph
