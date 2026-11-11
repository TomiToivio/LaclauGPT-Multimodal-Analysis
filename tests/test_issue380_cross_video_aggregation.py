"""Cross-video aggregation contract for Step 8 (#380).

The failure this guards against is quiet: aggregate the per-video tables the wrong way and a tie
observed once looks the same as a tie observed forty times, two different relation kinds become one
edge, and a metric computed over the merged soup is reported as if it described the political field.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET

from ep24_sna_aggregate import aggregate_edges, dedupe_events, export_view, views


def edge(
    *,
    event_id: str,
    item: str,
    source: str = "actor:a",
    target: str = "actor:b",
    relation: str = "quotes",
    evidence: str = "quoted in the thread",
    country: str = "finland",
    platform: str = "youtube",
    timestamp: str = "2026-10-01T10:00:00+00:00",
    source_label: str = "Actor A",
    target_label: str = "Actor B",
) -> dict:
    return {
        "edge_id": event_id,
        "canonical_item_id": item,
        "source": source,
        "target": target,
        "source_label": source_label,
        "target_label": target_label,
        "relation_type": relation,
        "directed": True,
        "evidence_quote": evidence,
        "country": country,
        "platform": platform,
        "timestamp": timestamp,
    }


def test_the_same_event_row_is_one_event_not_two():
    row = edge(event_id="edge:1", item="video:1")
    assert len(dedupe_events([row, dict(row)])) == 1
    nodes, ties = aggregate_edges([row, dict(row)])
    assert len(ties) == 1
    assert ties[0]["weight"] == 1, "a duplicated row must not double the weight"


def test_the_same_tie_in_three_videos_is_one_tie_of_weight_three():
    events = [
        edge(event_id="edge:1", item="video:1", timestamp="2026-10-01T10:00:00+00:00"),
        edge(event_id="edge:2", item="video:2", timestamp="2026-10-03T10:00:00+00:00"),
        edge(event_id="edge:3", item="video:3", timestamp="2026-10-02T10:00:00+00:00"),
    ]
    nodes, ties = aggregate_edges(events)
    assert len(ties) == 1
    tie = ties[0]
    assert tie["weight"] == 3
    assert tie["provenance_item_ids"] == ["video:1", "video:2", "video:3"]
    assert (tie["first_seen"], tie["last_seen"]) == (
        "2026-10-01T10:00:00+00:00",
        "2026-10-03T10:00:00+00:00",
    )
    assert {node["node_id"] for node in nodes} == {"actor:a", "actor:b"}


def test_relation_kinds_are_separate_views_and_the_combined_view_says_it_is_not_sna():
    events = [
        edge(event_id="edge:1", item="video:1", relation="quotes"),
        edge(event_id="edge:2", item="video:1", relation="replies"),
        edge(event_id="edge:3", item="video:2", relation="quotes", source="actor:a", target="actor:c"),
    ]
    result = views(events)
    assert set(result) == {"social:quotes", "social:replies", "combined"}
    assert result["social:quotes"]["is_one_mode_sna"] is True
    assert len(result["social:quotes"]["edges"]) == 2
    assert len(result["social:replies"]["edges"]) == 1
    assert result["combined"]["is_one_mode_sna"] is False


def test_metrics_are_recomputed_for_the_view_boundary():
    events = [
        edge(event_id="edge:1", item="video:1", country="finland",
             source="actor:fi-1", target="actor:fi-2"),
        edge(event_id="edge:2", item="video:2", country="poland",
             source="actor:pl-1", target="actor:pl-2"),
    ]
    everything = aggregate_edges(events)
    finland = aggregate_edges(events, country="finland")
    assert len(everything[1]) == 2 and len(finland[1]) == 1, "a weight across two countries describes neither"
    assert len(everything[0]) == 4 and len(finland[0]) == 2
    assert {node["node_id"] for node in finland[0]} == {"actor:fi-1", "actor:fi-2"}


def test_an_empty_corpus_is_a_view_not_a_crash():
    result = views([])
    assert result["combined"]["nodes"] == []
    assert result["combined"]["edges"] == []
    assert result["combined"]["metrics"]["edge_count"] == 0


def test_export_round_trips_and_is_deterministic(tmp_path):
    events = [
        edge(event_id="edge:1", item="video:1"),
        edge(event_id="edge:2", item="video:2", relation="replies", target="actor:c"),
    ]
    view = views(events)["combined"]
    first = export_view(view, tmp_path / "a", "combined")
    second = export_view(view, tmp_path / "b", "combined")

    for key in ("nodes_csv", "ties_csv", "json", "graphml"):
        assert first[key].read_bytes() == second[key].read_bytes(), f"{key} is not deterministic"

    with first["ties_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["provenance_item_ids"] == "video:1"

    payload = json.loads(first["json"].read_text(encoding="utf-8"))
    assert payload["is_one_mode_sna"] is False
    assert payload["metrics"]["edge_count"] == 2

    graph = ET.parse(first["graphml"]).getroot()
    namespace = "{http://graphml.graphdrawing.org/xmlns}"
    assert len(graph.findall(f"{namespace}graph/{namespace}node")) == 3, "a, b and c"
    assert len(graph.findall(f"{namespace}graph/{namespace}edge")) == 2
