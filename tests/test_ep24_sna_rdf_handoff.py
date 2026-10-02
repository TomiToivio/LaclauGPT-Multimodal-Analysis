"""Step 8 -> Step 9 handoff tests (issue #194, section 4).

The point of these is that the two stages share one contract: the tables
``roihu_sna.write_outputs`` produces are exactly what ``roihu_rdf`` projects, with
no ad-hoc translation in between. Synthetic fixtures only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import roihu_rdf as R
import roihu_sna as S


def rows() -> list[dict]:
    return [
        {
            "video_id": "v1", "country": "Finland", "source_type": "tiktok",
            "author_username": "acct_a",
            "allas_filename": "https://example.invalid/v1.mp4",
            "corrected_date": "2024-05-20",
            "entities": "Petteri Orpo, Kokkonen",
            "themes": "elections",
            "ep24_entity_resolution_json": json.dumps([
                {"surface_form": "Petteri Orpo", "canonical_name": "Petteri Orpo",
                 "entity_id": "FI-ORPO"},
                {"surface_form": "Kokkonen", "canonical_name": "Kokkonen",
                 "entity_id": "FI-KOK"},
            ]),
            "sna_edges_json": json.dumps([{
                "source_actor": "Petteri Orpo", "target_actor": "Kokkonen",
                "relation_type": "supports", "directed": True,
                "evidence_quote": "quoted", "confidence": 0.9,
            }]),
        },
        {"video_id": "v2", "country": "Finland", "source_type": "tiktok",
         "author_username": "acct_b", "allas_filename": "https://example.invalid/v2.mp4",
         "entities": "Petteri Orpo", "themes": "economy"},
    ]


@pytest.fixture()
def sna_tables(tmp_path: Path) -> Path:
    """Write SNA output exactly where roihu_rdf looks for it."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    network = S.build_network(rows(), country="Finland", language="fi")
    S.write_outputs(
        network, language="fi", country="Finland", output_dir=workdir / "sna",
        run_metadata={"run_id": "t"}, interpretation="stub",
    )
    return workdir


def test_step9_projects_step8_tables_without_translation(sna_tables: Path, monkeypatch):
    monkeypatch.chdir(sna_tables)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    found = R.maybe_emit_network(graph, "fi", prov)
    assert found["nodes"] > 0, "SNA nodes must reach the RDF export"
    assert found["edges"] > 0, "SNA edges must reach the RDF export"


def test_rdf_contains_sna_nodes_and_edges(sna_tables: Path, monkeypatch):
    monkeypatch.chdir(sna_tables)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    R.maybe_emit_network(graph, "fi", prov)
    turtle = graph.serialize()
    assert "SNANode" in turtle
    assert "SNAEdge" in turtle
    assert "sna/fi/" in turtle


def test_rdf_parses_with_rdflib(sna_tables: Path, monkeypatch):
    """The issue requires the produced RDF to parse."""
    rdflib = pytest.importorskip("rdflib")
    monkeypatch.chdir(sna_tables)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    R.maybe_emit_network(graph, "fi", prov)
    parsed = rdflib.Graph()
    parsed.parse(data=graph.serialize(), format="turtle")
    assert len(parsed) == graph.count()


def test_rdf_edge_uris_are_stable_across_runs(sna_tables: Path, monkeypatch):
    monkeypatch.chdir(sna_tables)
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    first, second = R.new_graph(), R.new_graph()
    R.maybe_emit_network(first, "fi", prov)
    R.maybe_emit_network(second, "fi", prov)
    assert first.serialize() == second.serialize(), "URIs must not vary per run"


def test_rdf_projects_node_metadata(sna_tables: Path, monkeypatch):
    monkeypatch.chdir(sna_tables)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    R.maybe_emit_network(graph, "fi", prov)
    turtle = graph.serialize()
    # country / platform / degree come from the Step 8 node table.
    assert "laclaugpt/country" in turtle.replace("lg:", "laclaugpt/") or "Finland" in turtle
    assert "tiktok" in turtle


def test_absent_sna_tables_skip_cleanly(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    found = R.maybe_emit_network(graph, "fi", prov)
    assert found == {"nodes": 0, "edges": 0}


def test_degenerate_node_table_is_skipped_not_emitted(tmp_path: Path, monkeypatch):
    """A table without ids must not produce meaningless nodes."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sna").mkdir()
    (tmp_path / "sna" / "sna_nodes_fi.csv").write_text("label\nsomething\n", encoding="utf-8")
    (tmp_path / "sna" / "sna_edges_fi.csv").write_text("source,target\n,\n", encoding="utf-8")
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    found = R.maybe_emit_network(graph, "fi", prov)
    assert found == {"nodes": 0, "edges": 0}


def test_no_secrets_leak_into_rdf(sna_tables: Path, monkeypatch):
    monkeypatch.chdir(sna_tables)
    graph = R.new_graph()
    prov = R.Provenance(run_id="t", model="m", generated_at="2026-01-01T00:00:00Z",
                        code_version="v")
    R.maybe_emit_network(graph, "fi", prov)
    blob = graph.serialize().casefold()
    for needle in ("password", "token", "secret", "api_key", "mongodb://", "redis://"):
        assert needle not in blob, needle
