"""Tests for the Step 8 SNA layer (issue #194).

Synthetic rows only. Public fixtures, no private research data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import roihu_sna as S

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def sample_rows() -> list[dict]:
    """Two items, one shared actor, one extracted stance relation."""
    return [
        {
            "video_id": "v1", "country": "Finland", "source_type": "tiktok",
            "author_username": "acct_a", "allas_filename": "https://example.invalid/v1.mp4",
            "corrected_date": "2024-05-20",
            "entities": "Petteri Orpo, Kokoomus",
            "new_entity": "Pääministeri Orpo",
            "themes": "elections, economy",
            "ep24_entity_resolution_json": json.dumps([
                {"surface_form": "Petteri Orpo", "normalized_form": "Petteri Orpo",
                 "canonical_name": "Petteri Orpo", "entity_id": "FI-ORPO", "variants": []},
                {"surface_form": "Pääministeri Orpo", "normalized_form": "Orpo",
                 "canonical_name": "Petteri Orpo", "entity_id": "FI-ORPO", "variants": []},
                {"surface_form": "Kokoomus", "canonical_name": "Kokoomus",
                 "entity_id": "FI-KOK", "variants": []},
            ]),
            "sna_edges_json": json.dumps([{
                "source_actor": "Petteri Orpo", "target_actor": "Kokoomus",
                "relation_type": "supports", "directed": True,
                "evidence_quote": "quoted evidence", "confidence": 0.9,
            }]),
        },
        {
            "video_id": "v2", "country": "Finland", "source_type": "tiktok",
            "author_username": "acct_b", "allas_filename": "https://example.invalid/v2.mp4",
            "entities": "Petteri Orpo", "themes": "economy",
            "ep24_entity_resolution_json": json.dumps([
                {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
                 "entity_id": "FI-ORPO", "variants": []},
            ]),
        },
    ]


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def test_build_network_produces_nodes_and_edges():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    assert len(net.nodes) > 0
    assert len(net.edges) > 0
    types = {n.node_type for n in net.nodes.values()}
    assert types == {"content", "account", "actor", "theme"}
    relations = {e.relation for e in net.edges.values()}
    assert {"posted", "mentions", "associated_with"} <= relations


def test_nodes_carry_type_label_provenance_and_country():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    for node in net.nodes.values():
        assert node.node_type, node
        assert node.label, node
        assert node.provenance, node
        assert node.country == "Finland", node


def test_edges_are_directed_with_source_target_and_evidence():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    for edge in net.edges.values():
        assert edge.source in net.nodes, edge
        assert edge.target in net.nodes, edge
        assert edge.evidence, edge
        assert edge.relation, edge


def test_content_node_uses_canonical_video_identity():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    contents = [n for n in net.nodes.values() if n.node_type == S.NODE_CONTENT]
    assert len(contents) == 2
    by_label = {n.label: n for n in contents}
    for node in contents:
        assert node.source_id == node.id
        assert node.source_url.startswith("https://example.invalid/")
    # A timestamp is carried through when the row has one, and left blank when it
    # does not -- absence must not be invented.
    assert by_label["v1"].timestamp == "2024-05-20"
    assert by_label["v2"].timestamp == ""


def test_identity_is_deterministic_across_runs():
    first = S.build_network(sample_rows(), country="Finland", language="fi")
    second = S.build_network(sample_rows(), country="Finland", language="fi")
    assert sorted(first.nodes) == sorted(second.nodes)
    assert sorted(first.edges) == sorted(second.edges)


def test_identity_does_not_depend_on_row_order():
    rows = sample_rows()
    forward = S.build_network(rows, country="Finland", language="fi")
    backward = S.build_network(list(reversed(rows)), country="Finland", language="fi")
    assert sorted(forward.nodes) == sorted(backward.nodes)


# --------------------------------------------------------------------------- #
# Identity -- the fragmentation bug this layer must not have
# --------------------------------------------------------------------------- #

def test_one_actor_is_one_node_across_surface_forms():
    """A registry-resolved actor must not fragment into several nodes."""
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    orpo = [n for n in net.nodes.values() if n.label.casefold().endswith("orpo")]
    # Both "Petteri Orpo" and the title-stripped mention resolve to the registry id.
    assert len(orpo) == 1, [(n.id, n.label) for n in orpo]
    assert orpo[0].id == "entity:FI-ORPO"


def test_bare_surname_without_records_groups_under_its_canonical_name():
    rows = [{
        "video_id": "v1", "country": "Finland", "entities": "Petteri Orpo",
        "new_entity": "Pääministeri Orpo",
        "ep24_entity_ids": '["FI-ORPO"]',
        "ep24_entity_canonical_names": '["Petteri Orpo"]',
    }]
    net = S.build_network(rows, country="Finland", language="fi")
    actors = [n for n in net.nodes.values() if n.node_type == S.NODE_ACTOR]
    assert len(actors) == 1, [(n.id, n.label) for n in actors]


def test_two_people_sharing_a_surname_stay_separate():
    rows = [{
        "video_id": "v3", "country": "Finland", "entities": "Petteri Orpo, Matti Orpo",
        "ep24_entity_resolution_json": json.dumps([
            {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo", "entity_id": "FI-P"},
            {"surface_form": "Matti Orpo", "canonical_name": "Matti Orpo", "entity_id": "FI-M"},
        ]),
    }]
    net = S.build_network(rows, country="Finland", language="fi")
    ids = {n.id for n in net.nodes.values() if n.node_type == S.NODE_ACTOR}
    assert ids == {"entity:FI-P", "entity:FI-M"}


def test_ambiguous_bare_surname_is_not_attributed_to_either_person():
    rows = [{
        "video_id": "v4", "country": "Finland", "entities": "Orpo",
        "ep24_entity_canonical_names": '["Matti Orpo", "Petteri Orpo"]',
    }]
    net = S.build_network(rows, country="Finland", language="fi")
    actor_ids = {n.id for n in net.nodes.values() if n.node_type == S.NODE_ACTOR}
    assert actor_ids, "the mention still yields a node"
    assert not any(i.endswith(("FI-P", "FI-M")) for i in actor_ids)


def test_flat_id_columns_are_never_zipped_positionally():
    """Regression: zipping ids with names independently-sorted mispaired people."""
    rows = [{
        "video_id": "v4", "country": "Finland", "entities": "Matti Orpo",
        "ep24_entity_ids": '["FI-PETTERI", "FI-MATTI"]',
        "ep24_entity_canonical_names": '["Matti Orpo", "Petteri Orpo"]',
    }]
    net = S.build_network(rows, country="Finland", language="fi")
    labels_by_id = {n.id: n.label for n in net.nodes.values() if n.node_type == S.NODE_ACTOR}
    # Whatever id is chosen, the label must not be paired with the OTHER person's id.
    assert "FI-PETTERI" not in labels_by_id, labels_by_id


def test_actor_is_one_node_across_rows_with_different_entity_coverage():
    """Regression: rows carrying registry records and rows without must not fragment.

    Row 1 has resolution records and yields ``entity:FI-ORPO``; row 2 has none and
    yielded a folded ``actor:<hash>`` node for the same person -- two nodes for one
    actor, visible only when different rows carry different amounts of entity output.
    """
    rows = [
        {"video_id": "v1", "country": "Finland", "author_username": "a",
         "entities": "Petteri Orpo",
         "ep24_entity_resolution_json": json.dumps([
             {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
              "entity_id": "FI-ORPO"}])},
        {"video_id": "v2", "country": "Finland", "author_username": "b",
         "entities": "Petteri Orpo"},
    ]
    net = S.build_network(rows, country="Finland", language="fi")
    actors = [n for n in net.nodes.values() if n.node_type == S.NODE_ACTOR]
    assert len(actors) == 1, [(n.id, n.label) for n in actors]
    assert actors[0].id == "entity:FI-ORPO"


def test_cross_row_reconciliation_never_merges_two_people():
    rows = [
        {"video_id": "v1", "country": "Finland", "entities": "Petteri Orpo",
         "ep24_entity_resolution_json": json.dumps([
             {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
              "entity_id": "FI-P"}])},
        {"video_id": "v2", "country": "Finland", "entities": "Matti Orpo"},
    ]
    net = S.build_network(rows, country="Finland", language="fi")
    ids = {n.id for n in net.nodes.values() if n.node_type == S.NODE_ACTOR}
    assert "entity:FI-P" in ids
    assert len(ids) == 2, ids


def test_reconciliation_drops_self_loops_it_creates():
    """Merging must not leave an edge pointing from a node to itself."""
    rows = [
        {"video_id": "v1", "country": "Finland", "entities": "Petteri Orpo",
         "ep24_entity_resolution_json": json.dumps([
             {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
              "entity_id": "FI-P"}])},
        {"video_id": "v2", "country": "Finland", "entities": "Petteri Orpo"},
    ]
    net = S.build_network(rows, country="Finland", language="fi")
    for edge in net.edges.values():
        assert edge.source != edge.target, edge


def test_reconciled_metrics_are_recomputed():
    rows = [
        {"video_id": "v1", "country": "Finland", "entities": "Petteri Orpo",
         "ep24_entity_resolution_json": json.dumps([
             {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
              "entity_id": "FI-P"}])},
        {"video_id": "v2", "country": "Finland", "entities": "Petteri Orpo"},
    ]
    net = S.build_network(rows, country="Finland", language="fi")
    for node in net.nodes.values():
        assert node.degree == node.in_degree + node.out_degree
    actor = next(n for n in net.nodes.values() if n.node_type == S.NODE_ACTOR)
    assert actor.degree == 2, "both mentions must be counted after the merge"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_basic_metrics_are_computed():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    for node in net.nodes.values():
        assert node.degree == node.in_degree + node.out_degree
        assert node.degree > 0, node
    summary = S.graph_summary(net)
    assert summary["nodes"] == len(net.nodes)
    assert summary["edges"] == len(net.edges)
    assert 0.0 <= summary["density"] <= 1.0
    assert summary["components"] >= 1


def test_density_matches_the_formula():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    n = len(net.nodes)
    expected = len(net.edges) / (n * (n - 1))
    assert S.density(net) == pytest.approx(expected)


def test_connected_components_groups_reachable_nodes():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    components = S.connected_components(net)
    assert sum(len(c) for c in components) == len(net.nodes)
    # Every node in one component must be reachable from the others.
    assert components[0], components


def test_weighted_degree_accumulates_repeated_edges():
    rows = [
        {"video_id": f"v{i}", "country": "Finland", "author_username": "same",
         "entities": "A"} for i in range(3)
    ]
    net = S.build_network(rows, country="Finland", language="fi")
    account = next(n for n in net.nodes.values() if n.node_type == S.NODE_ACCOUNT)
    assert account.weighted_degree == pytest.approx(3.0)
    assert account.out_degree == 3


def test_no_advanced_metrics_leak_in():
    """Scope guard: the issue says basic metrics only."""
    assert not hasattr(S, "community_detection")
    assert not hasattr(S, "betweenness")
    assert not hasattr(S, "eigenvector_centrality")


# --------------------------------------------------------------------------- #
# Castells interpretation
# --------------------------------------------------------------------------- #

def test_interpretation_is_separate_from_construction():
    """The theory layer reads a summary; it cannot add topology."""
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    before_nodes, before_edges = len(net.nodes), len(net.edges)
    summary = S.graph_summary(net)
    S.castells_interpretation(summary)
    S.castells_interpretation(summary, client=lambda prompt, system: "text")
    assert (len(net.nodes), len(net.edges)) == (before_nodes, before_edges)


def test_interpretation_uses_the_injected_client():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    seen: list[str] = []

    def client(prompt: str, system: str) -> str:
        seen.append(system)
        return "MODELLED INTERPRETATION"

    text = S.castells_interpretation(S.graph_summary(net), client=client)
    assert text == "MODELLED INTERPRETATION"
    assert seen and "Castells" in seen[0]
    # The system prompt must forbid fabricating topology.
    assert "never" in seen[0].casefold()


def test_interpretation_falls_back_without_a_model():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    text = S.castells_interpretation(S.graph_summary(net))
    assert text.strip()
    assert "interpretation" in text.casefold()


def test_broken_client_does_not_fail_the_stage():
    net = S.build_network(sample_rows(), country="Finland", language="fi")

    def broken(prompt: str, system: str) -> str:
        raise RuntimeError("model unavailable")

    text = S.castells_interpretation(S.graph_summary(net), client=broken)
    assert text.strip(), "interpretation must degrade, not crash"


def test_small_graph_caveat_is_stated():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    text = S.castells_interpretation(S.graph_summary(net))
    assert "caveat" in text.casefold() or "small" in text.casefold()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def test_report_has_the_documented_sections():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    report = S.render_report(net, country="Finland", language="fi")
    for heading in (
        "# Social Network Analysis", "## Dataset", "## Network construction",
        "## Nodes", "## Edges", "## Basic network statistics",
        "## Most connected nodes", "## Main observed relationships",
        "## Castellsian interpretation", "## Caveats and uncertainty",
        "## Provenance / run metadata",
    ):
        assert heading in report, heading


def test_report_marks_interpretation_as_interpretation():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    report = S.render_report(net, country="Finland", language="fi")
    section = report.split("## Castellsian interpretation")[1]
    assert "interpretation, not measurement" in section
    assert "theory" in section.casefold()


def test_report_is_prose_not_only_metrics():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    report = S.render_report(net, country="Finland", language="fi")
    construction = report.split("## Network construction")[1].split("## Nodes")[0]
    assert len(construction.split()) > 30, "construction section should explain in prose"


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def test_write_outputs_produces_tables_json_and_report(tmp_path: Path):
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    paths = S.write_outputs(net, language="fi", country="Finland", output_dir=tmp_path)
    for key in ("nodes_csv", "edges_csv", "network_json", "report_md"):
        assert Path(paths[key]).is_file(), key
    rows = list(csv.DictReader((tmp_path / "sna_nodes_fi.csv").open(encoding="utf-8")))
    assert rows and {"id", "node_type", "label"} <= set(rows[0])
    edge_rows = list(csv.DictReader((tmp_path / "sna_edges_fi.csv").open(encoding="utf-8")))
    assert edge_rows and {"source", "target", "relation"} <= set(edge_rows[0])


def test_written_tables_match_the_contract_roihu_rdf_reads(tmp_path: Path):
    """roihu_rdf.maybe_emit_network looks for ./sna/sna_{kind}s_<lang>.csv."""
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    S.write_outputs(net, language="fi", country="Finland", output_dir=tmp_path)
    assert (tmp_path / "sna_nodes_fi.csv").is_file()
    assert (tmp_path / "sna_edges_fi.csv").is_file()
    node_row = next(csv.DictReader((tmp_path / "sna_nodes_fi.csv").open(encoding="utf-8")))
    edge_row = next(csv.DictReader((tmp_path / "sna_edges_fi.csv").open(encoding="utf-8")))
    assert node_row["id"] and node_row["label"]
    assert edge_row["source"] and edge_row["target"]


def test_json_output_is_valid_and_complete(tmp_path: Path):
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    paths = S.write_outputs(net, language="fi", country="Finland", output_dir=tmp_path)
    payload = json.loads(Path(paths["network_json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == S.SCHEMA_VERSION
    assert len(payload["nodes"]) == len(net.nodes)
    assert len(payload["edges"]) == len(net.edges)
    assert payload["summary"]["nodes"] == len(net.nodes)


def test_no_secret_like_values_in_output(tmp_path: Path):
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    paths = S.write_outputs(net, language="fi", country="Finland", output_dir=tmp_path)
    blob = Path(paths["network_json"]).read_text(encoding="utf-8")
    for needle in ("password", "token", "secret", "api_key", "mongodb://"):
        assert needle not in blob.casefold()


# --------------------------------------------------------------------------- #
# Additive dataframe integration
# --------------------------------------------------------------------------- #

def test_append_to_dataframe_preserves_every_column():
    pd = pytest.importorskip("pandas")
    rows = sample_rows()
    frame = pd.DataFrame(rows)
    before = list(frame.columns)
    cells = list(frame["entities"])
    net = S.build_network(rows, country="Finland", language="fi")
    S.append_to_dataframe(frame, net)
    assert list(frame.columns)[: len(before)] == before
    assert list(frame["entities"]) == cells
    assert "sna_node_ids" in frame.columns


def test_append_to_dataframe_rejects_non_dataframe():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    with pytest.raises(TypeError):
        S.append_to_dataframe([{"a": 1}], net)


# --------------------------------------------------------------------------- #
# Empty / degenerate input
# --------------------------------------------------------------------------- #

def test_empty_input_yields_empty_graph():
    net = S.build_network([], country="Finland", language="fi")
    assert len(net.nodes) == 0
    assert len(net.edges) == 0
    summary = S.graph_summary(net)
    assert summary["nodes"] == 0
    assert summary["density"] == 0.0


def test_row_without_identifiers_still_produces_a_content_node():
    net = S.build_network([{"country": "Finland"}], country="Finland", language="fi")
    contents = [n for n in net.nodes.values() if n.node_type == S.NODE_CONTENT]
    assert len(contents) == 1


def test_extracted_edges_can_be_excluded():
    with_edges = S.build_network(sample_rows(), country="Finland", language="fi")
    without = S.build_network(
        sample_rows(), country="Finland", language="fi", include_extracted=False
    )
    assert len(without.edges) < len(with_edges.edges)


def test_stance_relations_are_never_invented():
    """Without an extracted edge, no supports/opposes relation may appear."""
    rows = [{"video_id": "v1", "country": "Finland", "entities": "A, B"}]
    net = S.build_network(rows, country="Finland", language="fi")
    assert not {e.relation for e in net.edges.values()} & {"supports", "opposes"}


def test_extracted_edge_keeps_its_evidence_quote():
    net = S.build_network(sample_rows(), country="Finland", language="fi")
    stance = [e for e in net.edges.values() if e.relation == "supports"]
    assert stance and stance[0].evidence_quote == "quoted evidence"
    assert stance[0].evidence == S.EVIDENCE_EXTRACTED
