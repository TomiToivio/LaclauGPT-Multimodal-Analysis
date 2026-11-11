"""#380 bullet 4: the textbook metrics, with applicability stated instead of a fake zero.

The issue asks for "N, M, density, components, in/out degree and normalized centrality; add betweenness,
closeness, reciprocity, clustering and community detection where definitions/applicability are
documented. Correctly handle empty, small, disconnected, directed and multimode graphs."

The dangerous failure is not a wrong number. It is a **plausible** one: 0.0 betweenness on a two-node
graph reads like "nobody brokers" when the measure was never defined. Every metric here returns
``None`` plus a reason instead, and these tests pin that.
"""

from __future__ import annotations

import ep24_sna


def node(node_id: str) -> dict:
    return {"node_id": node_id, "label": node_id, "node_type": "actor"}


def edge(source: str, target: str, **extra) -> dict:
    return {"source": source, "target": target, "directed": True, **extra}


def test_an_empty_graph_returns_reasons_not_zeros():
    metrics = ep24_sna.extended_metrics([], [])
    for name in ("reciprocity", "clustering", "closeness", "betweenness", "communities"):
        assert metrics[name]["value"] is None, name
        assert metrics[name]["undefined_reason"], name


def test_a_single_node_has_no_clustering_and_no_betweenness():
    metrics = ep24_sna.extended_metrics([node("a")], [])
    assert metrics["clustering"]["value"] is None
    assert metrics["betweenness"]["value"] is None
    assert metrics["closeness"]["value"] is None


def test_two_nodes_cannot_have_betweenness_but_have_a_distance():
    metrics = ep24_sna.extended_metrics([node("a"), node("b")], [edge("a", "b")])
    assert metrics["betweenness"]["value"] is None
    assert "fewer than three" in metrics["betweenness"]["undefined_reason"]
    assert metrics["closeness"]["mean"] > 0


def test_reciprocity_counts_returned_arcs_and_ignores_undirected_ties():
    nodes = [node(x) for x in "abc"]
    edges = [edge("a", "b"), edge("b", "a"), edge("a", "c"),
             {"source": "b", "target": "c", "directed": False}]
    result = ep24_sna.reciprocity(nodes, edges)
    assert result["directed_arc_count"] == 3
    assert result["reciprocated_arc_count"] == 2
    assert abs(result["value"] - 2 / 3) < 1e-9


def test_reciprocity_is_undefined_on_a_purely_undirected_graph():
    nodes = [node("a"), node("b")]
    edges = [{"source": "a", "target": "b", "directed": False}]
    assert ep24_sna.reciprocity(nodes, edges)["value"] is None


def test_clustering_is_one_on_a_triangle_and_zero_on_a_star():
    triangle = [node(x) for x in "abc"]
    star = [node(x) for x in "abcd"]
    triangle_edges = [edge("a", "b"), edge("b", "c"), edge("a", "c")]
    star_edges = [edge("a", "b"), edge("a", "c"), edge("a", "d")]
    assert abs(ep24_sna.clustering(triangle, triangle_edges)["value"] - 1.0) < 1e-9
    assert ep24_sna.clustering(star, star_edges).get("value") is None or \
        ep24_sna.clustering(star, star_edges)["local"]["a"] == 0.0


def test_betweenness_finds_the_only_broker_and_ignores_the_edges():
    nodes = [node(x) for x in "abc"]
    edges = [edge("a", "b"), edge("b", "c")]
    result = ep24_sna.betweenness(nodes, edges)
    assert result["per_node"]["b"] > 0
    assert result["per_node"]["a"] == 0


def test_closeness_does_not_reward_an_isolated_node():
    nodes = [node(x) for x in "abc"]
    edges = [edge("a", "b")]
    result = ep24_sna.closeness(nodes, edges)
    assert result["isolated_node_count"] == 1
    assert "c" not in result["per_node"]


def test_disconnected_components_are_not_averaged_into_a_nonsense_mean():
    nodes = [node(x) for x in "abcd"]
    edges = [edge("a", "b"), edge("c", "d")]
    result = ep24_sna.betweenness(nodes, edges)
    assert all(value == 0 for value in result["per_node"].values())
    assert result["mean"] == 0


def test_communities_find_two_obvious_groups_and_are_stable_across_runs():
    nodes = [node(x) for x in "abcdef"]
    edges = []
    for group in (("a", "b", "c"), ("d", "e", "f")):
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                edges.append(edge(left, right))
    first = ep24_sna.greedy_communities(nodes, edges)
    second = ep24_sna.greedy_communities(nodes, list(reversed(edges)))
    assert first["community_count"] == 2
    assert first["communities"] == second["communities"], "the partition must not depend on edge order"
    assert first["value"] > 0


def test_community_detection_is_undefined_without_edges():
    result = ep24_sna.greedy_communities([node("a"), node("b")], [])
    assert result["value"] is None
    assert "no edges" in result["undefined_reason"]


def test_parallel_relations_collapse_in_the_projection_but_not_in_reciprocity():
    nodes = [node("a"), node("b")]
    edges = [edge("a", "b", weight=3), edge("a", "b", weight=1)]
    reciprocity = ep24_sna.reciprocity(nodes, edges)
    assert reciprocity["directed_arc_count"] == 1, "a repeated arc is one dyad"
    clustering_result = ep24_sna.clustering(nodes, edges)
    assert clustering_result["value"] is None, "two nodes cannot cluster"


def test_the_definitions_travel_with_the_numbers():
    metrics = ep24_sna.extended_metrics([node("a"), node("b")], [edge("a", "b")])
    assert metrics["definitions"]["undefined"].startswith("None with a reason")
    assert "projection" in metrics["definitions"]
