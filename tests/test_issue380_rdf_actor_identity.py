"""Cross-stage canonical identity joins only corroborated entity IDs (#380)."""
import json

from roihu_csv_rdf import NS, canonical_actor_uri, project_row, uri


def test_unresolved_or_role_labels_never_canonicalize():
    base, project = "https://example.org/laclau", "ep24"
    assert canonical_actor_uri(base, project, "actor:raw-name") is None
    assert canonical_actor_uri(base, project, "Minister of Health") is None
    assert canonical_actor_uri(base, project, "") is None
    assert canonical_actor_uri(base, project, "entity:resolved-42") == uri(base, project, "actor", "resolved-42")


def test_dna_and_sna_share_only_resolved_actor_uri():
    base, project = "https://example.org/laclau", "ep24"
    row = {
        "_storage_id": "video-001", "country": "Finland",
        "dna_status": "ok",
        "dna_statements_json": json.dumps([{
            "statement_id": "s1", "actor_id": "entity:person-42",
            "actor_name_raw": "Alice", "concept": "Taxation", "agreement": 1,
        }, {
            "statement_id": "s2", "actor_id": "local-provisional",
            "actor_name_raw": "Minister", "concept": "Budget", "agreement": 0,
        }]),
        "sna_status": "ok",
        "sna_nodes_json": json.dumps([{
            "node_id": "entity:person-42", "label": "Alice", "node_type": "actor",
        }, {"node_id": "actor:raw-minister", "label": "Minister"}]),
        "sna_edges_json": json.dumps([{
            "edge_id": "e1", "source": "entity:person-42",
            "target": "actor:raw-minister", "relation_type": "quotation",
            "evidence_quote": "Alice quoted the minister.",
        }]),
    }
    _, _, graph, warnings = project_row(row, base=base, project=project, dataset="fi", row_number=1)
    assert warnings == []
    canonical = canonical_actor_uri(base, project, "entity:person-42")
    assert f"<{canonical}> <{NS}label> \"Alice\"" in graph
    assert f"<{uri(base, project, 'dna-statement', 's1')}> <{NS}dnaActor> <{canonical}>" in graph
    assert f"<{uri(base, project, 'sna-edge', 'e1')}> <{NS}source> <{canonical}>" in graph
    assert f"<{uri(base, project, 'dna-actor', 'local-provisional')}> <{NS}canonicalActor>" not in graph
    assert f"<{uri(base, project, 'sna-node', 'actor:raw-minister')}> <{NS}canonicalActor>" not in graph
