"""Deterministic graph construction and reporting for EP24 Step 8 SNA.

The LLM may extract evidence-supported relations, but this module owns graph
identity, topology, metrics and researcher-readable reporting. Keeping those
parts deterministic prevents a theoretical prompt from fabricating network
structure.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

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


def _token_set(text: str) -> set[str]:
    """Lowercased token set used only for conservative identity matching."""
    return {token for token in re.split(r"[\s,]+", text.casefold()) if token}


#: A token shorter than this may not be used for a subset match. Guards against a
#: degenerate subset such as "a" ⊂ "Anna Maria Something".
MIN_SUBSET_TOKEN_LENGTH = 3


def _title_stripped(text: str) -> str:
    """Title-stripped form, using the shared resolver's own vocabulary.

    Imported lazily and defensively: ``ep24_entities`` is a heavy module and this
    helper must not make ``ep24_sna`` unimportable if it changes shape. The
    returned type is checked because ``strip_titles`` returns ``(text, titles)``.
    """
    try:
        from ep24_entities import strip_titles
    except Exception:  # pragma: no cover - defensive
        return text.casefold().strip()
    try:
        result = strip_titles(text)
    except Exception:  # pragma: no cover - defensive
        return text.casefold().strip()
    if isinstance(result, tuple):
        result = result[0] if result else ""
    return str(result or "").casefold().strip()


def _fold(text: str) -> str:
    """Comparison fold, delegating to the shared resolver when available."""
    try:
        from ep24_entities import fold_key
    except Exception:  # pragma: no cover - defensive
        return text.casefold().strip()
    try:
        return fold_key(text)
    except Exception:  # pragma: no cover - defensive
        return text.casefold().strip()


def _merged_into(
    entity_id: str,
    canonical_name: str,
    label: str,
    *,
    min_token_length: int = MIN_SUBSET_TOKEN_LENGTH,
) -> str:
    """Return the merge reason when this mention provably denotes that entity.

    Three conditions, in decreasing strength, and each is deliberately narrow:

    1. **exact folded equality**;
    2. **title-stripped equality** — ``Pääministeri Orpo`` ≡ ``Orpo``;
    3. **one-directional token subset** — ``Orpo`` ⊂ ``Petteri Orpo``.

    Only the subset rule needs a guard: it requires every token of the mention to
    appear in the candidate's own label/alias tokens, and requires each matched
    token to be at least ``min_token_length`` long so a stray one- or two-character
    token cannot subset-match a long name.

    Returns ``""`` when nothing matches, so the caller cannot mistake a miss for a
    match by truthiness accident.
    """
    mention = _text(label)
    if not mention:
        return ""
    candidate_label = _text(canonical_name)
    mention_folded = _fold(mention)
    if mention_folded and mention_folded == _fold(candidate_label):
        return "exact_folded"
    mention_stripped = _title_stripped(mention)
    if mention_stripped and mention_stripped == _title_stripped(candidate_label):
        return "title_stripped"
    mention_tokens = _token_set(mention_stripped or mention)
    candidate_tokens = _token_set(_title_stripped(candidate_label) or candidate_label)
    if not mention_tokens or not candidate_tokens:
        return ""
    if not mention_tokens.issubset(candidate_tokens):
        return ""
    if any(len(token) < min_token_length for token in mention_tokens):
        return ""
    return "token_subset"


def reconcile_actor_nodes(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    row: Mapping[str, Any],
    *,
    min_token_length: int = MIN_SUBSET_TOKEN_LENGTH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Merge unresolved actor nodes into resolved entities — conservatively.

    The defect this fixes: ``node_identity`` prefers the canonical entity id, but
    Step 8's enrichment only supplies one when the mention matches a record in
    ``ep24_entity_resolution_json`` *by exact folded label*. A bare surname
    (``Orpo``), a title-prefixed form (``Pääministeri Orpo``) or an inflected form
    therefore becomes its own ``actor:<hash>`` node **even though the row contains a
    RESOLVED record for that person** — splitting one actor and inflating
    ``node_count``, ``density``, ``degree`` and ``component_count``.

    Policy (issue #197, author-specified): a ``actor:<hash>`` node is merged into a
    resolved ``entity:<id>`` node **only when exactly one** resolved candidate in
    the row matches. Zero candidates ⇒ the node stays separate. Two or more ⇒ the
    node stays separate **and is counted**, because two people can share a surname
    and guessing between them is the failure this layer exists to prevent.

    Returns ``(nodes, edges, report)``. Every merge is recorded on the surviving
    node as ``merged_from`` / ``merge_reason`` so it is auditable and reversible,
    and edges are **rewritten, never dropped**.
    """
    node_rows = [dict(node) for node in nodes]
    edge_rows = [dict(edge) for edge in edges]

    resolved: dict[str, str] = {}
    canonical_by_id: dict[str, str] = {}
    try:
        from ep24_entities import resolution_lookup

        for key, item in resolution_lookup(row.get("ep24_entity_resolution_json")).items():
            entity_id = _text(item.get("entity_id"))
            if not entity_id:
                continue
            resolved[key] = entity_id
            # The candidate set is the ROW'S resolved records, not the nodes the
            # edges happened to create. A row can resolve two people who share a
            # surname while only one of them is mentioned by an edge; building the
            # candidates from nodes alone would then see a single candidate and
            # merge the bare surname into one of them — exactly the false merge the
            # issue's `exactly one` rule exists to prevent. Measured on a fixture
            # with two resolved Kowalskas, node-based candidates produced
            # ambiguous_count == 0 and merged the surname.
            canonical_by_id[entity_id] = _text(item.get("canonical_name"))
    except Exception:  # pragma: no cover - defensive
        resolved = {}
        canonical_by_id = {}

    # A canonical name may be absent from the lookup item; fall back to the node
    # label for that entity so matching still has something to compare against.
    for node in node_rows:
        node_id = _text(node.get("node_id"))
        if not node_id.startswith("entity:"):
            continue
        entity_id = node_id.split(":", 1)[1]
        if entity_id not in canonical_by_id or not canonical_by_id[entity_id]:
            canonical_by_id[entity_id] = _text(node.get("label"))
    canonical_by_id = {
        (f"entity:{eid}" if not eid.startswith("entity:") else eid): name
        for eid, name in canonical_by_id.items()
    }

    report: dict[str, Any] = {
        "merged": [],
        "ambiguous": [],
        "unresolved": [],
    }

    # -- decide per unresolved node -----------------------------------------
    row_item_id = canonical_item_id(row)
    row_country = value(row, "country")
    row_platform = _platform(row)
    row_source_url = _source_url(row)
    row_timestamp = _timestamp(row)
    existing_node_ids = {_text(n.get("node_id")) for n in node_rows}

    remap: dict[str, str] = {}
    for node in node_rows:
        node_id = _text(node.get("node_id"))
        if not node_id or node_id.startswith("entity:"):
            continue
        label = _text(node.get("label"))
        matches: list[tuple[str, str]] = []
        for entity_id, canonical_name in canonical_by_id.items():
            reason = _merged_into(
                entity_id, canonical_name, label, min_token_length=min_token_length
            )
            if reason:
                matches.append((entity_id, reason))
        # de-duplicate by target, keeping the strongest (first-listed) reason
        by_target: dict[str, str] = {}
        for entity_id, reason in matches:
            by_target.setdefault(entity_id, reason)
        if len(by_target) == 1:
            target, reason = next(iter(by_target.items()))
            # The target may come from the row's records while having no node of its
            # own (the edge never mentioned it directly). A node must exist before an
            # edge can point at it, so materialise one from the row's canonical name
            # rather than emitting an edge to a nonexistent endpoint.
            if target not in existing_node_ids:
                existing_node_ids.add(target)
                synthetic = {
                    "node_id": target,
                    "node_type": "actor",
                    "label": canonical_by_id.get(target, ""),
                    "country": str(row_country),
                    "platform": str(row_platform),
                    "source_url": str(row_source_url),
                    "canonical_item_id": str(row_item_id),
                    "timestamp": str(row_timestamp),
                    "provenance_item_ids": [str(row_item_id)] if row_item_id else [],
                    "merged_from": [],
                    "merge_reason": [],
                    "materialised_from": "row_resolution_record",
                }
                node_rows.append(synthetic)
            remap[node_id] = target
            report["merged"].append({
                "merged_from": node_id,
                "merged_into": target,
                "label": label,
                "merge_reason": reason,
            })
        elif len(by_target) > 1:
            report["ambiguous"].append({
                "node_id": node_id,
                "label": label,
                "candidates": sorted(by_target),
            })
        else:
            report["unresolved"].append({"node_id": node_id, "label": label})

    # -- apply the remap ----------------------------------------------------
    # Nodes the remap targets are inserted FIRST so the surviving node is the
    # canonical one. Otherwise a plain `actor:` node that happens to remap to the
    # same id can be processed before a materialised target and win the
    # first-writer-takes-all insert, silently dropping the materialisation marker
    # and the row's canonical label.
    ordered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in node_rows:
        node_id = _text(node.get("node_id"))
        if node_id in remap:
            continue
        ordered.append(node)
        seen_ids.add(node_id)
    for node in node_rows:
        node_id = _text(node.get("node_id"))
        if node_id in remap and node_id not in seen_ids:
            ordered.append(node)
            seen_ids.add(node_id)

    survivors: dict[str, dict[str, Any]] = {}
    for node in ordered:
        node_id = _text(node.get("node_id"))
        target = remap.get(node_id, node_id)
        existing = survivors.get(target)
        if existing is None:
            kept = {**node, "node_id": target}
            # Copy the provenance list rather than sharing the caller's. Within one
            # call every node comes from the same row and carries the same item id,
            # so the dedupe below masks a shared list today -- but the function must
            # not depend on that, and a shared list would leak a mutation into the
            # caller's node dicts the moment the ids differ.
            kept["provenance_item_ids"] = list(node.get("provenance_item_ids") or [])
            kept.setdefault("merged_from", [])
            kept.setdefault("merge_reason", [])
            survivors[target] = kept
            continue
        # fold provenance and record the merge on the surviving node
        for item_id in node.get("provenance_item_ids", []) or []:
            if item_id and item_id not in existing["provenance_item_ids"]:
                existing["provenance_item_ids"].append(item_id)
        if node_id != target:
            if node_id not in existing["merged_from"]:
                existing["merged_from"].append(node_id)
            reason = next(
                (m["merge_reason"] for m in report["merged"] if m["merged_from"] == node_id),
                "unknown",
            )
            if reason not in existing["merge_reason"]:
                existing["merge_reason"].append(reason)

    merged_nodes = list(survivors.values())

    # -- rewrite edges onto the surviving nodes (never drop them) -----------
    rewritten: list[dict[str, Any]] = []
    for edge in edge_rows:
        source = remap.get(_text(edge.get("source")), _text(edge.get("source")))
        target = remap.get(_text(edge.get("target")), _text(edge.get("target")))
        item = {**edge, "source": source, "target": target}
        if "source_label" in item and source != _text(edge.get("source")):
            item["source_label"] = survivors[source].get("label", item.get("source_label"))
        if "target_label" in item and target != _text(edge.get("target")):
            item["target_label"] = survivors[target].get("label", item.get("target_label"))
        provenance = dict(item.get("provenance") or {})
        if _text(edge.get("source")) != source or _text(edge.get("target")) != target:
            provenance["reconciled_from"] = {
                "source": _text(edge.get("source")),
                "target": _text(edge.get("target")),
            }
            item["provenance"] = provenance
        rewritten.append(item)

    report["merged_count"] = len(report["merged"])
    report["ambiguous_count"] = len(report["ambiguous"])
    report["unresolved_count"] = len(report["unresolved"])
    return merged_nodes, rewritten, report


def identity_fragmentation(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Report how much of the graph is *still* un-resolved identity (#197).

    The reconciliation pass merges what it can prove; whatever it cannot must be
    **visible in the numbers** rather than silently shaping `node_count`, `density`
    and `degree`. An unresolved actor is a node that is not backed by a canonical
    entity id; ambiguous ones were left separate on purpose.
    """
    node_rows = list(nodes)
    unresolved = [
        {"node_id": _text(node.get("node_id")), "label": _text(node.get("label"))}
        for node in node_rows
        if _text(node.get("node_type")) == "actor"
        and not _text(node.get("node_id")).startswith("entity:")
    ]
    return {
        "unresolved_actor_count": len(unresolved),
        "unresolved_actors": unresolved,
    }


def apply_reconciliation_metrics(
    metrics: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach the reconciliation and fragmentation counters to a metrics mapping.

    Kept separate from ``basic_metrics`` so the metric values themselves stay a pure
    function of the graph, and so a caller that never reconciles is unaffected.
    """
    updated = dict(metrics)
    updated["reconciled_actor_count"] = int((report or {}).get("merged_count", 0))
    updated["ambiguous_actor_count"] = int((report or {}).get("ambiguous_count", 0))
    updated["unresolved_actor_count"] = int((report or {}).get("unresolved_count", 0))
    updated["ambiguous_actors"] = list((report or {}).get("ambiguous", []))
    return updated


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
            # Issue #197: remaining fragmentation must be visible in the summary, not
            # silently shaping the counts above.
            f"reconciled_actors={metrics.get('reconciled_actor_count', 0)}",
            f"ambiguous_actors={metrics.get('ambiguous_actor_count', 0)}",
            f"unresolved_actors={metrics.get('unresolved_actor_count', 0)}",
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
