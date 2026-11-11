"""Read-only, relation-specific cross-video SNA views for downstream clients.

Only resolved entity IDs join across documents. Provisional names stay local.
The event table is authoritative; projection weights count distinct events.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from ep24_sna import basic_metrics


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _rows(value: Any) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value or "[]")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise ValueError("Expected a JSON array of graph records")
    return value


def _identity(raw: str, document: str) -> str:
    if raw.startswith("entity:") and raw != "entity:":
        return raw
    digest = hashlib.sha256(json.dumps([document, raw]).encode()).hexdigest()
    return "provisional:" + digest


def aggregate_social_events(
    records: Iterable[Mapping[str, Any]], *, relation: str,
    countries: Iterable[str] = (), platforms: Iterable[str] = (),
    start: str | None = None, end: str | None = None,
    actors: Iterable[str] = (), min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Select one relation and recompute metrics on the selected event boundary.

    Dates are inclusive ISO-8601 bounds. Missing dates/confidence are excluded
    when their corresponding filter is requested. Actor selection keeps incident
    events (an ego view), not an induced subgraph. No source records are changed.
    Conflicting copies of an event ID fail closed rather than picking a winner.
    """
    if not relation or not 0 <= min_confidence <= 1:
        raise ValueError("A relation and confidence in [0, 1] are required")
    lower, upper = _time(start) if start else None, _time(end) if end else None
    if lower and upper and lower > upper:
        raise ValueError("start must not follow end")
    countries, platforms, actors = set(countries), set(platforms), set(actors)
    events: dict[str, dict] = {}
    nodes: dict[str, dict] = {}
    rejected = 0
    for row in records:
        document = str(row.get("_storage_id") or "").strip()
        if not document:
            raise ValueError("Every record requires a stable _storage_id")
        labels = {str(n.get("node_id")): n.get("label", "")
                  for n in _rows(row.get("sna_nodes_json"))}
        for raw in _rows(row.get("sna_edges_json")):
            if raw.get("relation_type") != relation:
                continue
            country = str(raw.get("country") or row.get("country") or "")
            platform = str(raw.get("platform") or row.get("platform") or row.get("source_type") or "")
            if countries and country not in countries or platforms and platform not in platforms:
                continue
            timestamp = str(raw.get("timestamp") or "")
            if lower or upper:
                try:
                    date = _time(timestamp)
                except ValueError:
                    rejected += 1
                    continue
                if lower and date < lower or upper and date > upper:
                    continue
            try:
                confidence = float(raw.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = -1
            if not 0 <= confidence <= 1 or confidence < min_confidence:
                rejected += 1
                continue
            event_id = str(raw.get("edge_id") or "").strip()
            source, target = str(raw.get("source") or ""), str(raw.get("target") or "")
            evidence = str(raw.get("evidence_quote") or "").strip()
            directed = raw.get("directed", True)
            if not event_id or not source or not target or not evidence or not isinstance(directed, bool):
                rejected += 1
                continue
            if raw.get("canonical_item_id") not in (None, "", document):
                raise ValueError("Event provenance disagrees with its source document")
            source, target = _identity(source, document), _identity(target, document)
            if actors and not actors.intersection((source, target)):
                continue
            event = {**raw, "edge_id": event_id, "source": source, "target": target,
                     "canonical_item_id": document, "country": country,
                     "platform": platform, "timestamp": timestamp,
                     "directed": directed, "confidence": confidence,
                     "layer": "step8_social", "evidence_quote": evidence,
                     "model_prompt_provenance": {
                         key: row[key] for key in ("sna_prompt_version", "sna_schema_version", "sna_model", "_pipeline")
                         if key in row}}
            if event_id in events and events[event_id] != event:
                raise ValueError("Conflicting copies of an event ID")
            events[event_id] = event
            for node_id, original in ((source, raw["source"]), (target, raw["target"])):
                node = nodes.setdefault(node_id, {"node_id": node_id, "labels": set(), "provenance_item_ids": set()})
                node["labels"].add(str(labels.get(original) or original))
                node["provenance_item_ids"].add(document)
    event_rows = [events[key] for key in sorted(events)]
    node_rows = [{"node_id": key, "labels": sorted(nodes[key]["labels"]),
                  "provenance_item_ids": sorted(nodes[key]["provenance_item_ids"])} for key in sorted(nodes)]
    pairs: dict[tuple[str, str], dict] = {}
    for event in event_rows:
        orientations = [(event["source"], event["target"])]
        if not event["directed"] and event["source"] != event["target"]:
            orientations.append((event["target"], event["source"]))
        for source, target in orientations:
            pair = pairs.setdefault((source, target), {"source": source, "target": target,
                                    "relation_type": relation, "event_ids": set(), "source_item_ids": set()})
            pair["event_ids"].add(event["edge_id"])
            pair["source_item_ids"].add(event["canonical_item_id"])
    projection = [{**pairs[key], "event_ids": sorted(pairs[key]["event_ids"]),
                   "source_item_ids": sorted(pairs[key]["source_item_ids"]),
                   "weight": len(pairs[key]["event_ids"])} for key in sorted(pairs)]
    metrics = basic_metrics(node_rows, event_rows)
    metrics["components"] = sorted(metrics["components"])
    return {"schema_version": "ep24-social-view/1", "boundary": {
        "relation": relation, "countries": sorted(countries), "platforms": sorted(platforms),
        "start": start, "end": end, "actors": sorted(actors), "min_confidence": min_confidence,
        "nodes": "endpoints of selected events; no unobserved isolates",
        "projection": "directed; undirected events become two arcs; weights count unique events",
    }, "nodes": node_rows, "events": event_rows, "projection": projection,
        "metrics": metrics, "rejected_events": rejected}
