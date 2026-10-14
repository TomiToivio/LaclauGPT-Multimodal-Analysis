"""Step 9 must project discourse statements separately from SNA edges."""
from roihu_csv_rdf import NS, project_row


def test_step9_dna_and_typed_sna():
    import json
    row = {
        "_storage_id": "ep24-fi-test",
        "country": "finland",
        "dna_status": "ok",
        "dna_statements_json": json.dumps([{
            "statement_id": "s1", "organization": "Kokoomus",
            "concept": "EU membership", "agreement": 1,
            "evidence_quote": "EU-jäsenyys", "network_layer": "discourse",
        }, {
            "statement_id": "s2", "organization": "Critic",
            "concept": "Budget cuts", "agreement": 0,
        }]),
        "sna_status": "ok",
        "sna_nodes_json": json.dumps([
            {"node_id": "actor:a", "label": "Iltalehti"},
            {"node_id": "actor:b", "label": "Stubb"},
        ]),
        "sna_edges_json": json.dumps([{
            "edge_id": "e1", "source": "actor:a", "target": "actor:b",
            "relation_type": "mention", "network_layer": "communication",
            "evidence_quote": "Stubb kunnioitti Suomen kaatuneita.",
        }]),
    }
    _, _, triples, warnings = project_row(
        row, base="https://data.example/laclaugpt", project="ep24",
        dataset="test", row_number=1)
    assert not warnings
    assert f"<{NS}DNAStatement>" in triples
    assert f"<{NS}dnaStatement>" in triples
    assert '"support"' in triples and '"oppose"' in triples
    assert '"communication"' in triples
    assert '"EU membership"' in triples


def test_step9_ignores_failed_network_projections():
    import json
    row = {
        "_storage_id": "failed",
        "dna_status": "error",
        "dna_statements_json": json.dumps([{"organization": "A", "concept": "B"}]),
        "sna_status": "error",
        "sna_edges_json": json.dumps([{"edge_id": "x", "source": "a",
                                       "target": "b", "edge_type": "mention"}]),
    }
    _, _, triples, _ = project_row(
        row, base="https://data.example/laclaugpt", project="ep24",
        dataset="test", row_number=1)
    assert f"<{NS}DNAStatement>" not in triples
    assert f"<{NS}SNAEdge>" not in triples
