"""Regression tests for the October 2026 Step 8 audit."""
from types import SimpleNamespace

from ep24_sna import basic_metrics, graph_from_edges, render_markdown_report
from step_8_roihu_social_network_analysis import _network_relation_gate


def test_multigraph_density_counts_unique_ordered_pairs():
    nodes = [{"node_id": "a"}, {"node_id": "b"}]
    edges = [
        {"source": "a", "target": "b", "directed": True},
        {"source": "a", "target": "b", "directed": True},
        {"source": "a", "target": "b", "directed": True},
    ]
    metrics = basic_metrics(nodes, edges)
    assert metrics["edge_count"] == 3
    assert metrics["unique_directed_pair_count"] == 1
    assert metrics["density"] == 0.5
    assert metrics["simple_directed_density"] == 0.5
    assert metrics["mean_relations_per_pair"] == 3.0


def test_report_does_not_duplicate_llm_sections():
    row = {"_storage_id": "item-1", "country": "FI"}
    metrics = basic_metrics([], [])
    report = render_markdown_report(
        row, [], [], metrics, "DISTINCT EMPIRICAL REPORT",
        "DISTINCT CASTELLS REPORT",
    )
    assert "DISTINCT EMPIRICAL REPORT" not in report
    assert "DISTINCT CASTELLS REPORT" not in report
    assert "Unique directed actor pairs" in report


def test_attributed_speech_is_not_verbatim_quotation():
    edge = SimpleNamespace(
        source_actor="Reporter", target_actor="Minister",
        relation_type="quotation",
        evidence_quote="The minister said that the decision was final.",
    )
    accepted = _network_relation_gate([edge])
    assert len(accepted) == 1
    assert edge.relation_type == "reported_statement"


def test_ideological_relationship_still_rejected():
    edge = SimpleNamespace(
        source_actor="Party", target_actor="EU",
        relation_type="support", evidence_quote="We support membership.",
    )
    assert _network_relation_gate([edge]) == []


def test_reported_speaker_is_not_a_second_actor():
    row = {"_storage_id": "story", "country": "FI"}
    raw = [
        {"source_actor": "Li Andersson", "target_actor": "Reporter",
         "relation_type": "mention", "evidence_quote": "Li Andersson was cited."},
        {"source_actor": "Reporter", "target_actor": "Li Andersson (as reported by speaker)",
         "relation_type": "reported_statement", "evidence_quote": "Li Andersson said that."},
    ]
    nodes, edges = graph_from_edges(raw, row)
    li_nodes = [n for n in nodes if n["label"] == "Li Andersson"]
    assert len(li_nodes) == 1
    assert len(nodes) == 2
    assert edges[0]["source"] == edges[1]["target"]
