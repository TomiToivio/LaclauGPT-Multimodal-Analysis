"""Issue #194: deterministic SNA graph contract and RDF handoff tests."""

import json

from ep24_sna import basic_metrics, graph_from_edges, render_markdown_report
from roihu_csv_rdf import NS, project_row

ROW = {
    "country": "Finland",
    "author_username": "account-a",
    "source_type": "TikTok",
    "video_id": "000194",
    "allas_filename": "ep24/finland/000194.mp4",
    "entities": '["Alice", "Bob"]',
    "themes": '["EU policy"]',
}


def _edge():
    return {
        "source_actor": "Alice",
        "source_actor_id": "person-alice",
        "source_actor_canonical_name": "Alice Example",
        "target_actor": "Bob",
        "target_actor_id": "person-bob",
        "target_actor_canonical_name": "Bob Example",
        "relation_type": "mentions",
        "directed": True,
        "evidence_quote": "Alice explicitly mentions Bob.",
        "confidence": 0.9,
        "network_layer": "social",
    }


def test_sna_builds_stable_nodes_edges_and_basic_metrics():
    nodes_a, edges_a = graph_from_edges([_edge()], ROW)
    nodes_b, edges_b = graph_from_edges([_edge()], ROW)
    assert nodes_a == nodes_b
    assert edges_a == edges_b
    assert len(nodes_a) == 2
    assert len(edges_a) == 1
    assert nodes_a[0]["canonical_item_id"]
    assert edges_a[0]["source"].startswith("entity:")
    assert edges_a[0]["target"].startswith("entity:")
    assert edges_a[0]["edge_id"].startswith("edge:")

    metrics = basic_metrics(nodes_a, edges_a)
    assert metrics["node_count"] == 2
    assert metrics["edge_count"] == 1
    assert metrics["component_count"] == 1
    assert metrics["density"] == 0.5


def test_sna_drops_relations_without_evidence_instead_of_inventing_graph():
    bad = dict(_edge())
    bad["evidence_quote"] = ""
    nodes, edges = graph_from_edges([bad], ROW)
    assert nodes == []
    assert edges == []
    assert basic_metrics(nodes, edges)["edge_count"] == 0


def test_human_report_separates_empirical_graph_and_castells_interpretation():
    nodes, edges = graph_from_edges([_edge()], ROW)
    metrics = basic_metrics(nodes, edges)
    report = render_markdown_report(
        ROW,
        nodes,
        edges,
        metrics,
        "Alice explicitly mentions Bob.",
        "This may be read as a visible communication flow; no programmer or switcher role can be established.",
    )
    assert "# Social Network Analysis" in report
    assert "## Nodes" in report
    assert "## Edges" in report
    assert "## Basic network statistics" in report
    assert "## Castellsian interpretation" in report
    assert "theory prompt cannot add nodes or edges" in report
    assert "programmer or switcher" in report


def test_rdf_projects_same_sna_node_and_edge_identity_with_provenance():
    nodes, edges = graph_from_edges([_edge()], ROW)
    metrics = basic_metrics(nodes, edges)
    row = {
        **ROW,
        "sna_nodes_json": json.dumps(nodes),
        "sna_edges_json": json.dumps(edges),
        "sna_metrics_json": json.dumps(metrics),
    }
    _, _, graph, warnings = project_row(
        row,
        base="https://example.org/laclaugpt",
        project="ep24",
        dataset="finland",
        row_number=1,
    )
    assert warnings == []
    assert NS + "SNANode" in graph
    assert NS + "SNAEdge" in graph
    assert NS + "source" in graph
    assert NS + "target" in graph
    assert NS + "relationType" in graph
    assert "person-alice" in graph
    assert "person-bob" in graph
    assert "Alice explicitly mentions Bob." in graph
    assert NS + "SNAMetrics" in graph


def test_rdf_keeps_account_entities_themes_and_raw_cells():
    _, _, graph, warnings = project_row(
        ROW,
        base="https://example.org/laclaugpt",
        project="ep24",
        dataset="finland",
        row_number=1,
    )
    assert warnings == []
    assert NS + "Account" in graph
    assert NS + "Entity" in graph
    assert NS + "Theme" in graph
    assert "account-a" in graph
    assert "Alice" in graph
    assert "EU policy" in graph
    assert NS + "CSVCell" in graph
