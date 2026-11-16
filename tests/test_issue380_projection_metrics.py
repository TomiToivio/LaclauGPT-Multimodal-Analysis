"""Regression checks for textbook directed simple projection (#380).

Event multiplicity is retained in the multigraph, but a simple projection
must not interpret parallel observations as additional adjacent actors.
"""
from ep24_sna import basic_metrics


def test_parallel_events_do_not_inflate_normalized_simple_degree():
    nodes = [{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}]
    edges = [{"source": "a", "target": "b", "directed": True}] * 4
    result = basic_metrics(nodes, edges)
    assert result["edge_count"] == 4
    assert result["simple_directed_density"] == 1 / 6
    assert result["simple_out_degree"]["a"] == 1
    assert result["simple_out_degree_centrality"]["a"] == 0.5
    assert result["simple_in_degree_centrality"]["b"] == 0.5
    assert result["simple_directed_reciprocity"] == 0


def test_undirected_event_projects_to_two_arcs_and_reciprocity():
    nodes = [{"node_id": "a"}, {"node_id": "b"}]
    result = basic_metrics(nodes, [
        {"source": "a", "target": "b", "directed": False},
    ])
    assert result["unique_directed_pair_count"] == 2
    assert result["simple_directed_density"] == 1.0
    assert result["simple_directed_reciprocity"] == 1.0
    assert result["simple_in_degree_centrality"] == {"a": 1.0, "b": 1.0}
    assert result["simple_out_degree_centrality"] == {"a": 1.0, "b": 1.0}


def test_empty_singleton_and_self_loops_do_not_divide_by_zero():
    assert basic_metrics([], [])["simple_directed_reciprocity"] == 0
    result = basic_metrics(
        [{"node_id": "a"}], [{"source": "a", "target": "a"}],
    )
    assert result["simple_directed_density"] == 0
    assert result["simple_in_degree_centrality"]["a"] == 0
    assert result["simple_out_degree_centrality"]["a"] == 0


def test_reciprocity_counts_both_directions_without_duplicate_inflation():
    nodes = [{"node_id": k} for k in "abc"]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "a"},
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
    ]
    result = basic_metrics(nodes, edges)
    assert result["unique_directed_pair_count"] == 3
    assert result["simple_directed_reciprocity"] == 2 / 3
