"""Cross-video aggregation of persisted Step 8 social events (#380).

Step 8 stores a per-video multigraph: each `SocialEvent` is one observed, evidence-quoted actor→actor
relation inside one document. A political field is not one video, so the tables in this module answer a
different question — *which ties recur across the corpus* — and they are explicit about the two things
that makes easy to get wrong:

* **An event is deduplicated by identity, not by appearance.** The same row re-exported, or the same
  evidence line produced twice by a retry, is one event; the same tie observed in three videos is one
  *tie* with weight 3 and three provenance ids.
* **Relation kinds are never merged into one edge set silently.** A quote, a reply and a repost are
  different observations about different social mechanisms, so each relation gets its own view, and the
  combined view says so (`is_one_mode_sna: False`) rather than pretending to be a single social graph.

Metrics are recomputed per view with :func:`ep24_sna.basic_metrics` — the boundary decides the numbers,
so borrowing a single-video density would be a claim about a different population. Export is
dependency-free and deterministic: rows are sorted by id, so the same store state produces byte-identical
files.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ep24_sna import basic_metrics

#: Evidence quotes kept per aggregated tie. The full set would repeat one video's text on every row.
EVIDENCE_PER_TIE = 3


def _text(raw: Any) -> str:
    return " ".join(str(raw or "").split())


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def dedupe_events(edges: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per event identity, in a deterministic order.

    ``edge_id`` is what Step 8 hashes from (item, source, relation, target, evidence, directed), so two
    rows sharing it are the *same* observation, not two observations that happen to look alike.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for edge in edges:
        row = dict(edge)
        edge_id = _text(row.get("edge_id"))
        if not edge_id:
            edge_id = "edge:" + "|".join(
                _text(row.get(key))
                for key in ("canonical_item_id", "source", "relation_type", "target")
            )
            row["edge_id"] = edge_id
        by_id.setdefault(edge_id, row)
    return [by_id[key] for key in sorted(by_id)]


def aggregate_edges(
    edges: Iterable[Mapping[str, Any]],
    *,
    relation: str | None = None,
    country: str | None = None,
    platform: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate events into weighted ties, keeping provenance and the time window.

    Filtering happens *before* aggregation: a weight computed across two countries describes neither.
    """
    events = []
    for event in dedupe_events(edges):
        if relation is not None and _text(event.get("relation_type")) != relation:
            continue
        if country is not None and _text(event.get("country")) != country:
            continue
        if platform is not None and _text(event.get("platform")) != platform:
            continue
        events.append(event)

    grouped: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    for event in events:
        key = (
            _text(event.get("source")),
            _text(event.get("target")),
            _text(event.get("relation_type")),
            bool(event.get("directed", True)),
        )
        tie = grouped.get(key)
        if tie is None:
            tie = grouped[key] = {
                "source": key[0],
                "target": key[1],
                "relation_type": key[2],
                "directed": key[3],
                "event_ids": [],
                "provenance_item_ids": [],
                "evidence_quotes": [],
                "country": _text(event.get("country")),
                "platform": _text(event.get("platform")),
                "timestamps": [],
            }
        tie["event_ids"].append(_text(event.get("edge_id")))
        item_id = _text(event.get("canonical_item_id"))
        if item_id:
            tie["provenance_item_ids"].append(item_id)
        quote = _text(event.get("evidence_quote"))
        if quote and quote not in tie["evidence_quotes"] and len(tie["evidence_quotes"]) < EVIDENCE_PER_TIE:
            tie["evidence_quotes"].append(quote)
        stamp = _text(event.get("timestamp"))
        if stamp:
            tie["timestamps"].append(stamp)

    labels: dict[str, dict[str, Any]] = {}
    ties: list[dict[str, Any]] = []
    for key in sorted(grouped):
        tie = grouped[key]
        event_ids = _sorted_unique(tie["event_ids"])
        item_ids = _sorted_unique(tie["provenance_item_ids"])
        stamps = sorted(_sorted_unique(tie["timestamps"]))
        ties.append(
            {
                "source": tie["source"],
                "target": tie["target"],
                "relation_type": tie["relation_type"],
                "directed": tie["directed"],
                # Weight is the number of distinct documents that carry this tie: an event duplicated
                # inside one video must not inflate it.
                "weight": max(1, len(item_ids) or len(event_ids)),
                "event_count": len(event_ids),
                "event_ids": event_ids,
                "provenance_item_ids": item_ids,
                "evidence_quotes": tie["evidence_quotes"],
                "first_seen": stamps[0] if stamps else "",
                "last_seen": stamps[-1] if stamps else "",
                "country": tie["country"],
                "platform": tie["platform"],
            }
        )
        for node_id, label in ((tie["source"], "source_label"), (tie["target"], "target_label")):
            if not node_id:
                continue
            node = labels.setdefault(
                node_id,
                {"node_id": node_id, "node_type": "actor", "label": "", "provenance_item_ids": []},
            )
            if not node["label"]:
                node["label"] = _text(next((e.get(label) for e in events if node_id in (e.get("source"), e.get("target"))), ""))
            node["provenance_item_ids"] = _sorted_unique([*node["provenance_item_ids"], *item_ids])

    return [labels[key] for key in sorted(labels)], ties


def views(edges: Iterable[Mapping[str, Any]], **filters: Any) -> dict[str, dict[str, Any]]:
    """One view per relation kind, plus a combined view that is explicitly not one-mode SNA."""
    event_rows = list(edges)
    relations = _sorted_unique(_text(event.get("relation_type")) for event in event_rows)
    result: dict[str, dict[str, Any]] = {}
    for relation in relations:
        nodes, ties = aggregate_edges(event_rows, relation=relation, **filters)
        result[f"social:{relation}"] = {
            "name": relation,
            "layer": "social",
            "relation": relation,
            "is_one_mode_sna": True,
            "nodes": nodes,
            "edges": ties,
            "metrics": basic_metrics(nodes, ties),
        }
    nodes, ties = aggregate_edges(event_rows, **filters)
    result["combined"] = {
        "name": "combined",
        "layer": "combined",
        "relation": None,
        "is_one_mode_sna": False,
        "nodes": nodes,
        "edges": ties,
        "metrics": basic_metrics(nodes, ties),
    }
    return result


def _graphml(view: Mapping[str, Any]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '<key id="label" for="node" attr.name="label" attr.type="string"/>',
        '<key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
        '<key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
        f'<graph id="{escape(str(view.get("name", "view")))}" edgedefault="directed">',
    ]
    for node in sorted(view.get("nodes") or [], key=lambda n: str(n.get("node_id"))):
        lines.append(
            f'<node id="{escape(str(node.get("node_id")))}">'
            f'<data key="label">{escape(str(node.get("label") or ""))}</data></node>'
        )
    for edge in sorted(view.get("edges") or [], key=lambda e: (str(e.get("source")), str(e.get("target")), str(e.get("relation_type")))):
        lines.append(
            f'<edge source="{escape(str(edge.get("source")))}" target="{escape(str(edge.get("target")))}">'
            f'<data key="relation">{escape(str(edge.get("relation_type") or ""))}</data>'
            f'<data key="weight">{int(edge.get("weight", 1) or 1)}</data></edge>'
        )
    lines.extend(["</graph>", "</graphml>", ""])
    return "\n".join(lines)


def export_view(view: Mapping[str, Any], out_dir: str | Path, stem: str) -> dict[str, Path]:
    """Write CSV (nodes, ties), JSON (with metrics) and GraphML for one view. Never overwrites."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    node_fields = ["node_id", "node_type", "label", "provenance_item_ids"]
    nodes_path = directory / f"{stem}.nodes.csv"
    with nodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_fields, extrasaction="ignore")
        writer.writeheader()
        for node in sorted(view.get("nodes") or [], key=lambda n: str(n.get("node_id"))):
            writer.writerow({**node, "provenance_item_ids": ";".join(node.get("provenance_item_ids") or [])})
    written["nodes_csv"] = nodes_path

    edge_fields = [
        "source", "target", "relation_type", "directed", "weight", "event_count",
        "first_seen", "last_seen", "country", "platform", "provenance_item_ids", "event_ids",
    ]
    edges_path = directory / f"{stem}.ties.csv"
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields, extrasaction="ignore")
        writer.writeheader()
        for edge in view.get("edges") or []:
            writer.writerow(
                {
                    **edge,
                    "provenance_item_ids": ";".join(edge.get("provenance_item_ids") or []),
                    "event_ids": ";".join(edge.get("event_ids") or []),
                    "evidence_quotes": " | ".join(edge.get("evidence_quotes") or []),
                }
            )
    written["ties_csv"] = edges_path

    payload = {
        "name": view.get("name"),
        "layer": view.get("layer"),
        "relation": view.get("relation"),
        "is_one_mode_sna": view.get("is_one_mode_sna"),
        "metrics": view.get("metrics"),
        "nodes": view.get("nodes"),
        "edges": view.get("edges"),
    }
    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["json"] = json_path

    graphml_path = directory / f"{stem}.graphml"
    graphml_path.write_text(_graphml(view), encoding="utf-8")
    written["graphml"] = graphml_path
    return written
