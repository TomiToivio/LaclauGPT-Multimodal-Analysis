"""Cross-video, cross-country graph views without inventing a network (#380).

The acceptance criterion this file pins is the one that is easy to fake: *"multi-video and
multi-country graph views and filters available with typed links, exact provenance and sensible
scalable rendering."* A module that concatenates per-video edge lists and calls the result a network
would satisfy the sentence and break the methodology — density across videos, an actor–concept coding
edge summed with a mention, a weight with no source behind it.

So the tests are about what the aggregation refuses to do: it will not merge relation families, it
will not compute one-mode metrics across layers, it will not count the same event twice, and it will
not turn a Step 7 statement into a social tie.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ep24_graph_aggregate as agg


def event(**overrides):
    base = {
        "layer": agg.LAYER_STEP8,
        "edge_id": "e1",
        "source": "actor:a",
        "target": "actor:b",
        "source_label": "A",
        "target_label": "B",
        "relation": "mention",
        "directed": True,
        "country": "finland",
        "platform": "youtube",
        "item_id": "v1",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def statement(**overrides):
    base = {
        "statement_id": "s1",
        "organization": "Acme Institute",
        "actor_id": "ORG1",
        "concept": "AI risk",
        "concept_id": "C1",
        "agreement": "agree",
        "document_id": "v1",
        "date_time": "2026-01-02T00:00:00Z",
        "evidence_quote": "we agree",
    }
    base.update(overrides)
    return base


def test_one_event_is_counted_once_even_when_exported_twice():
    doubled = agg.dedupe_events([event(), event()])
    assert len(doubled) == 1
    assert agg.aggregate_ties(doubled)[0]["weight"] == 1.0


def test_repeated_ties_aggregate_with_their_provenance():
    ties = agg.aggregate_ties(
        [
            event(edge_id="e1", item_id="v1", timestamp="2026-01-01T00:00:00Z"),
            event(edge_id="e2", item_id="v2", timestamp="2026-02-01T00:00:00Z", platform="x"),
        ]
    )
    assert len(ties) == 1
    tie = ties[0]
    assert tie["weight"] == 2.0 and tie["event_count"] == 2
    assert tie["item_ids"] == ["v1", "v2"]
    assert tie["countries"] == ["finland"] and tie["platforms"] == ["x", "youtube"]
    assert tie["first_seen"].startswith("2026-01-01") and tie["last_seen"].startswith("2026-02-01")


def test_relation_families_are_never_summed():
    """Two mentions and one quotation are two ties, not a weight of three."""
    ties = agg.aggregate_ties(
        [
            event(edge_id="e1"),
            event(edge_id="e2"),
            event(edge_id="e3", relation="quotation"),
        ]
    )
    by_relation = {tie["relation"]: tie["weight"] for tie in ties}
    assert by_relation == {"mention": 2.0, "quotation": 1.0}


def test_a_step7_statement_is_an_actor_concept_edge_not_a_social_tie():
    edges = agg.dna_edges([statement()])
    assert len(edges) == 1
    edge = edges[0]
    assert edge["layer"] == agg.LAYER_STEP7
    assert edge["target_type"] == "concept" and edge["source_type"] == "actor"
    assert edge["relation"] == "agree"


def test_namesakes_are_not_merged_by_label():
    """Two organisations that both call themselves "Acme" stay two nodes unless an id says otherwise."""
    edges = agg.dna_edges(
        [
            statement(statement_id="s1", actor_id="ORG1"),
            statement(statement_id="s2", actor_id="ORG2"),
            statement(statement_id="s3", actor_id=""),
        ]
    )
    nodes = {edge["source"] for edge in edges}
    assert len(nodes) == 3, nodes


def test_combined_view_is_heterogeneous_and_refuses_one_mode_metrics():
    view = agg.build_view(
        [event(edge_id="e1"), event(edge_id="e2", relation="quotation")],
        agg.dna_edges([statement()]),
        view=agg.VIEW_COMBINED,
    )
    metrics = view["metrics"]
    assert view["heterogeneous"] is True
    assert metrics["metric_status"] == "not_applicable"
    assert "heterogeneous multilayer" in metrics["reason"]
    # ... but each layer on its own is still measured, and labelled
    assert set(metrics["layer_metrics"]) == {agg.LAYER_STEP7, agg.LAYER_STEP8}
    assert metrics["layer_metrics"][agg.LAYER_STEP8]["metric_status"] == "computed"


def test_a_single_layer_view_computes_metrics_on_the_simple_projection():
    view = agg.build_view(
        [
            event(edge_id="e1"),
            event(edge_id="e2"),
            event(edge_id="e3", relation="quotation"),
            event(edge_id="e4", source="actor:b", target="actor:a"),
        ],
        view=agg.VIEW_STEP8,
    )
    metrics = view["metrics"]
    assert metrics["metric_status"] == "computed"
    assert metrics["node_count"] == 2
    assert metrics["edge_count"] == 2, "two ordered pairs: a->b and b->a"
    assert metrics["density"] == pytest.approx(1.0)
    assert "simple directed actor pair" in metrics["projection"]
    # the composite arc still says which relation families it is made of
    arc = [row for row in view["edges"] if row["source"] == "actor:a"][0]
    assert arc["relations"] == ["mention", "quotation"]
    assert arc["weight"] == 3.0


def test_filters_select_by_country_platform_relation_actor_and_period():
    ties = agg.aggregate_ties(
        [
            event(edge_id="e1", country="finland", platform="youtube"),
            event(edge_id="e2", country="poland", platform="x", relation="quotation"),
            event(edge_id="e3", source="actor:c", target="actor:d", relation="reply", platform="tiktok"),
        ]
    )
    assert [t["relation"] for t in agg.apply_filters(ties, countries=["poland"])] == ["quotation"]
    assert [t["relation"] for t in agg.apply_filters(ties, platforms=["youtube"])] == ["mention"]
    assert [t["relation"] for t in agg.apply_filters(ties, platforms=["tiktok"])] == ["reply"]
    assert [t["relation"] for t in agg.apply_filters(ties, relations=["reply"])] == ["reply"]
    assert [t["relation"] for t in agg.apply_filters(ties, actors=["actor:c"])] == ["reply"]
    assert agg.apply_filters(ties, countries=["narnia"]) == []


def test_period_filter_uses_the_ties_own_window():
    ties = agg.aggregate_ties(
        [
            event(edge_id="e1", timestamp="2026-01-01T00:00:00Z"),
            event(edge_id="e2", timestamp="2026-06-01T00:00:00Z"),
        ]
    )
    kept = agg.apply_filters(ties, since="2026-05-01")
    assert len(kept) == 1 and kept[0]["last_seen"].startswith("2026-06-01")


def test_exports_are_readable_and_carry_the_layer(tmp_path: Path):
    view = agg.build_view(
        [event(edge_id="e1"), event(edge_id="e2", relation="quotation")],
        agg.dna_edges([statement()]),
        view=agg.VIEW_COMBINED,
    )
    written = agg.write_view_bundle(view, tmp_path, stem="demo")

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["view"] == "combined" and payload["tie_count"] == 3

    with open(written["edges_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["layer"] for row in rows} == {agg.LAYER_STEP7, agg.LAYER_STEP8}
    assert {row["relation"] for row in rows} == {"mention", "quotation", "agree"}

    graphml = ET.parse(written["graphml"]).getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    edge_layers = {
        data.text for data in graphml.findall(".//g:edge/g:data[@key='layer']", ns)
    }
    assert edge_layers == {agg.LAYER_STEP7, agg.LAYER_STEP8}
    assert len(graphml.findall(".//g:node", ns)) == len(view["nodes"])

    gexf = ET.parse(written["gexf"]).getroot()
    gexf_ns = {"x": "http://gexf.net/1.2"}
    assert len(gexf.findall(".//x:node", gexf_ns)) == len(view["nodes"])
    assert len(gexf.findall(".//x:edge", gexf_ns)) == len(view["ties"])


def test_step8_export_reader_accepts_the_file_the_pipeline_writes(tmp_path: Path):
    path = tmp_path / "finland.sna_edges.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "edge_id",
                "source",
                "target",
                "source_label",
                "target_label",
                "relation_type",
                "directed",
                "canonical_item_id",
                "country",
                "platform",
                "timestamp",
                "evidence_quote",
                "confidence",
                "network_layer",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "edge_id": "edge:1",
                "source": "actor:a",
                "target": "actor:b",
                "source_label": "A",
                "target_label": "B",
                "relation_type": "mention",
                "directed": "True",
                "canonical_item_id": "v1",
                "country": "finland",
                "platform": "youtube",
                "timestamp": "2026-01-01T00:00:00Z",
                "evidence_quote": "@B said this",
                "confidence": "0.8",
                "network_layer": "communication",
            }
        )
    edges = agg.load_step8_edges([path])
    assert len(edges) == 1
    assert edges[0]["layer"] == agg.LAYER_STEP8
    assert edges[0]["relation"] == "mention"
    assert edges[0]["confidence"] == pytest.approx(0.8)
    assert edges[0]["directed"] is True


def test_cli_writes_a_bundle_from_persisted_inputs(tmp_path: Path):
    edges_csv = tmp_path / "finland.sna_edges.csv"
    edges_csv.write_text(
        "edge_id,source,target,relation_type,directed,canonical_item_id,country,platform,timestamp\n"
        "edge:1,actor:a,actor:b,mention,True,v1,finland,youtube,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    dna_json = tmp_path / "dna.json"
    dna_json.write_text(json.dumps([statement()]), encoding="utf-8")

    exit_code = agg.main(
        [
            "--step8-edges",
            str(edges_csv),
            "--dna-json",
            str(dna_json),
            "--view",
            "combined",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "out" / "combined.json").exists()
    assert (tmp_path / "out" / "combined.graphml").exists()
