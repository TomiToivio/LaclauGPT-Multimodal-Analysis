"""Deterministic graph construction and reporting for EP24 Step 8 SNA.

The LLM may extract evidence-supported relations, but this module owns graph
identity, topology, metrics and researcher-readable reporting. Keeping those
parts deterministic prevents a theoretical prompt from fabricating network
structure.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from ep24_schema import stable_source_id, value

SNA_SCHEMA_VERSION = "ep24-sna/1.0"


def _text(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _hash(*parts: str) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_item_id(row: Mapping[str, Any]) -> str:
    existing = _text(row.get("_storage_id"))
    return existing or stable_source_id(row)


def _platform(row: Mapping[str, Any]) -> str:
    return value(row, "source_type") or _text(row.get("platform"))


def _source_url(row: Mapping[str, Any]) -> str:
    for key in ("source_url", "url", "webVideoUrl", "video_url"):
        text = _text(row.get(key))
        if text:
            return text
    return ""


def _timestamp(row: Mapping[str, Any]) -> str:
    for key in ("videoCreated", "created_at", "timestamp", "published_at"):
        text = _text(row.get(key))
        if text:
            return text
    return ""


def node_identity(label: str, *, entity_id: str = "", node_type: str = "actor") -> str:
    """Return stable node identity preferring upstream entity resolution."""
    entity_id = _text(entity_id)
    if entity_id:
        return f"entity:{entity_id}"
    return f"{node_type}:{_hash(node_type, label.casefold())}"


def graph_from_edges(
    raw_edges: Iterable[Mapping[str, Any]],
    row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize extracted relations into explicit node and edge tables."""
    item_id = canonical_item_id(row)
    country = value(row, "country")
    platform = _platform(row)
    source_url = _source_url(row)
    timestamp = _timestamp(row)

    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for raw in raw_edges:
        source_label = _text(
            raw.get("source_actor_canonical_name") or raw.get("source_actor")
        )
        target_label = _text(
            raw.get("target_actor_canonical_name") or raw.get("target_actor")
        )
        relation = _text(raw.get("relation_type"))
        evidence = _text(raw.get("evidence_quote"))
        if not source_label or not target_label or not relation or not evidence:
            continue

        source_id = node_identity(
            source_label,
            entity_id=_text(raw.get("source_actor_id")),
            node_type="actor",
        )
        target_id = node_identity(
            target_label,
            entity_id=_text(raw.get("target_actor_id")),
            node_type="actor",
        )

        for node_id, label in ((source_id, source_label), (target_id, target_label)):
            node = nodes_by_id.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "node_type": "actor",
                    "label": label,
                    "country": country,
                    "platform": platform,
                    "source_url": source_url,
                    "canonical_item_id": item_id,
                    "timestamp": timestamp,
                    "provenance_item_ids": [],
                },
            )
            if item_id and item_id not in node["provenance_item_ids"]:
                node["provenance_item_ids"].append(item_id)

        directed = bool(raw.get("directed", True))
        edge_id = "edge:" + _hash(
            item_id, source_id, relation, target_id, evidence, str(directed)
        )
        edge = {
            **dict(raw),
            "edge_id": edge_id,
            "source": source_id,
            "target": target_id,
            "source_label": source_label,
            "target_label": target_label,
            "edge_type": relation,
            "relation_type": relation,
            "directed": directed,
            "network_layer": _text(raw.get("network_layer")) or "social",
            "canonical_item_id": item_id,
            "country": country,
            "platform": platform,
            "source_url": source_url,
            "timestamp": timestamp,
            "provenance": {
                "canonical_item_id": item_id,
                "source_url": source_url,
                "country": country,
                "platform": platform,
                "timestamp": timestamp,
            },
        }
        edges.append(edge)

    return list(nodes_by_id.values()), edges


def basic_metrics(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Small, inspectable metrics set; components treat directed ties as weak ties."""
    node_rows = list(nodes)
    edge_rows = list(edges)
    node_ids = {str(node["node_id"]) for node in node_rows if node.get("node_id")}
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    weighted_degree = defaultdict(int)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    directed_edges = 0

    for edge in edge_rows:
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        if source not in node_ids or target not in node_ids:
            continue
        directed = bool(edge.get("directed", True))
        weight_raw = edge.get("weight", 1)
        try:
            weight = max(1, int(float(weight_raw)))
        except (TypeError, ValueError):
            weight = 1
        out_degree[source] += 1
        in_degree[target] += 1
        weighted_degree[source] += weight
        weighted_degree[target] += weight
        if not directed:
            out_degree[target] += 1
            in_degree[source] += 1
        else:
            directed_edges += 1
        adjacency[source].add(target)
        adjacency[target].add(source)

    degree = {
        node_id: int(in_degree[node_id] + out_degree[node_id])
        for node_id in node_ids
    }

    components: list[list[str]] = []
    unseen = set(node_ids)
    while unseen:
        start = next(iter(unseen))
        queue = deque([start])
        unseen.remove(start)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    n = len(node_ids)
    unique_arcs = {
        (_text(edge.get("source")), _text(edge.get("target")))
        for edge in edge_rows
        if _text(edge.get("source")) in node_ids and _text(edge.get("target")) in node_ids
    }
    density = (len(unique_arcs) / (n * (n - 1))) if n > 1 else 0.0
    centrality = {
        node_id: (degree[node_id] / (2 * (n - 1)) if n > 1 else 0.0)
        for node_id in node_ids
    }

    return {
        "schema_version": SNA_SCHEMA_VERSION,
        "node_count": n,
        "edge_count": len(edge_rows),
        "directed_edge_count": directed_edges,
        "density": density,
        "component_count": len(components),
        "components": components,
        "degree": dict(sorted(degree.items())),
        "in_degree": dict(sorted(in_degree.items())),
        "out_degree": dict(sorted(out_degree.items())),
        "weighted_degree": dict(sorted(weighted_degree.items())),
        "degree_centrality": dict(sorted(centrality.items())),
    }


def graph_summary(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> str:
    node_rows = list(nodes)
    edge_rows = list(edges)
    labels = {str(n.get("node_id")): _text(n.get("label")) for n in node_rows}
    degree = metrics.get("degree", {})
    top = sorted(
        ((int(score), labels.get(node_id, node_id)) for node_id, score in degree.items()),
        reverse=True,
    )[:10]
    relations = [
        f"- {e.get('source_label')} --{e.get('edge_type')}--> {e.get('target_label')} "
        f"[evidence: {e.get('evidence_quote')}]"
        for e in edge_rows[:50]
    ]
    return "\n".join(
        [
            f"nodes={metrics.get('node_count', 0)}",
            f"edges={metrics.get('edge_count', 0)}",
            f"density={float(metrics.get('density', 0.0)):.6f}",
            f"weak_components={metrics.get('component_count', 0)}",
            "most_connected=" + json.dumps(top, ensure_ascii=False),
            "relations:",
            *(relations or ["- <none>"]),
        ]
    )


def render_markdown_report(
    row: Mapping[str, Any],
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    empirical_summary: str,
    castells_interpretation: str,
) -> str:
    node_rows = list(nodes)
    edge_rows = list(edges)
    item_id = canonical_item_id(row)
    generated_at = datetime.now(timezone.utc).isoformat()
    labels = {str(n.get("node_id")): _text(n.get("label")) for n in node_rows}
    ranked = sorted(
        (
            (int(score), labels.get(node_id, node_id))
            for node_id, score in metrics.get("degree", {}).items()
        ),
        reverse=True,
    )

    node_lines = [
        f"- `{node.get('node_id')}` ({node.get('node_type')}): {node.get('label')}"
        for node in node_rows
    ] or ["- No evidence-supported nodes extracted."]
    edge_lines = [
        f"- {edge.get('source_label')} → **{edge.get('edge_type')}** → "
        f"{edge.get('target_label')} (evidence: “{edge.get('evidence_quote')}”)"
        for edge in edge_rows
    ] or ["- No evidence-supported edges extracted."]
    top_lines = [
        f"- {label}: degree {score}" for score, label in ranked[:10]
    ] or ["- No connected nodes."]

    empirical = _text(empirical_summary) or (
        "No additional model-written empirical summary was produced. "
        "The node/edge tables above remain the authoritative graph representation."
    )
    castells = _text(castells_interpretation) or (
        "No Castellsian interpretation was produced. The graph can still be used "
        "as a descriptive network artifact without theoretical interpretation."
    )

    return "\n".join(
        [
            "# Social Network Analysis",
            "",
            "## Dataset",
            f"- Canonical item ID: `{item_id}`",
            f"- Country: {value(row, 'country') or '<unknown>'}",
            f"- Platform/source type: {_platform(row) or '<unknown>'}",
            f"- Source URL: {_source_url(row) or '<not supplied>'}",
            "",
            "## Network construction",
            "Only relations explicitly supported by upstream evidence are admitted. "
            "The graph topology is deterministic after extraction; the theory prompt cannot add nodes or edges.",
            "",
            "## Nodes",
            *node_lines,
            "",
            "## Edges",
            *edge_lines,
            "",
            "## Basic network statistics",
            f"- Nodes: {metrics.get('node_count', 0)}",
            f"- Edges: {metrics.get('edge_count', 0)}",
            f"- Directed edges: {metrics.get('directed_edge_count', 0)}",
            f"- Weakly connected components: {metrics.get('component_count', 0)}",
            f"- Density: {float(metrics.get('density', 0.0)):.6f}",
            "",
            "## Most connected nodes",
            *top_lines,
            "",
            "## Main observed relationships",
            empirical,
            "",
            "## Castellsian interpretation",
            castells,
            "",
            "## Caveats and uncertainty",
            "This is a descriptive network built from the analysed sample, not evidence of hidden "
            "coordination, causality, friendship, strategic control or population-level influence. "
            "Terms such as programmer, switcher and network-making power are interpretive concepts "
            "and should be used only when the observed relations justify them.",
            "",
            "## Provenance / run metadata",
            f"- SNA schema: `{SNA_SCHEMA_VERSION}`",
            f"- Generated at: {generated_at}",
            f"- Canonical item ID: `{item_id}`",
        ]
    )
