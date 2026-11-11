import copy
import json

import pytest

from ep24_sna_aggregate import aggregate_social_events


def record(item, source="entity:a", target="entity:b", country="finland", **extra):
    return {"_storage_id": item, "country": country, "sna_edges_json": [{
        "edge_id": item + ":event", "source": source, "target": target,
        "relation_type": "mention", "evidence_quote": "A mentions B",
        "timestamp": "2026-01-02T00:00:00Z", "confidence": .9, **extra}]}


def test_dedup_weights_and_read_only_input():
    rows = [record("v1"), record("v2")]
    before = copy.deepcopy(rows)
    result = aggregate_social_events(rows + rows, relation="mention")
    assert rows == before
    assert len(result["events"]) == 2
    assert result["projection"][0]["weight"] == 2
    assert result["projection"][0]["source_item_ids"] == ["v1", "v2"]
    assert result["metrics"]["density"] == .5
    assert json.dumps(result, sort_keys=True) == json.dumps(aggregate_social_events(reversed(rows), relation="mention"), sort_keys=True)


def test_unresolved_names_are_document_local():
    result = aggregate_social_events([record("v1", target="actor:minister"), record("v2", target="actor:minister")], relation="mention")
    assert len(result["nodes"]) == 3


def test_filters_recompute_boundary_and_exclude_other_relations():
    rows = [record("v1"), record("v2", target="entity:c", country="poland"), record("v3", relation_type="affiliation")]
    result = aggregate_social_events(rows, relation="mention", countries=["finland"], start="2026-01-01", end="2026-01-03")
    assert len(result["events"]) == 1
    assert result["metrics"]["node_count"] == 2
    assert result["metrics"]["density"] == .5
    assert aggregate_social_events(rows, relation="reply")["metrics"]["node_count"] == 0


def test_conflicting_ids_and_provenance_fail_closed():
    with pytest.raises(ValueError, match="Conflicting"):
        aggregate_social_events([record("v1"), record("v1", target="entity:c")], relation="mention")
    with pytest.raises(ValueError, match="provenance"):
        aggregate_social_events([record("v1", canonical_item_id="other")], relation="mention")


def test_missing_evidence_and_invalid_direction_rejected():
    result = aggregate_social_events([record("v1", evidence_quote=""), record("v2", directed="false")], relation="mention")
    assert result["rejected_events"] == 2
    assert result["events"] == []


def test_undirected_event_has_two_projection_arcs_not_two_events():
    result = aggregate_social_events([record("v1", directed=False)], relation="mention")
    assert len(result["events"]) == 1
    assert len(result["projection"]) == 2
    assert result["metrics"]["simple_directed_reciprocity"] == 1


def test_export_roundtrip_and_no_overwrite(tmp_path):
    import hashlib

    import networkx as nx

    from scripts.ep24.export_social_network import export_view
    view = aggregate_social_events([record("v1"), record("v2")], relation="mention")
    output = tmp_path / "private"
    export_view(view, output)
    assert json.loads((output / "network.json").read_text()) == view
    for name in ("events.graphml", "events.gexf"):
        graph = nx.read_graphml(output / name) if name.endswith("graphml") else nx.read_gexf(output / name)
        assert graph.number_of_edges() == 2
        assert graph.number_of_nodes() == 2
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
        assert (output / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export_view(view, output)
