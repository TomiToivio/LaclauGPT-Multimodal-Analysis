#!/usr/bin/env python3
"""Basic, inspectable Social Network Analysis for the EP24 pipeline (issue #194).

Deliberately modest, per the issue's scope note: **Node – Edge – Node**, a small
set of interpretable metrics, and a human-readable report. Advanced SNA
(community detection, temporal/multiplex networks) is explicitly out of scope.

Design rules that matter more than the code:

* **Empirical construction is separate from theoretical interpretation.** The
  graph is built only from relationships the input actually supports. The
  Castells lens is applied afterwards, to a summarised graph, and is labelled as
  interpretation wherever it appears.
* **The theory never generates topology.** ``castells_interpretation`` reads a
  graph summary; it cannot add nodes or edges. A test asserts this.
* **Stable canonical identities.** Node ids are derived from the canonical
  content identity and the entity identity, reusing ``ep24_entities`` helpers
  rather than inventing a parallel scheme. No random or per-run ids.
* **Additive.** Every incoming column is preserved; SNA results are appended.

The output contract is CSV nodes/edges tables plus JSON, written under ``sna/``
as ``sna_nodes_<language>.csv`` / ``sna_edges_<language>.csv``. That path and the
``id`` / ``label`` / ``source`` / ``target`` column names are what
``roihu_rdf.maybe_emit_network`` already looks for, so the RDF export consumes
these without an ad-hoc adapter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ep24_entities import fold_key, strip_titles
from ep24_schema import value as ep24_value

LOG = logging.getLogger(__name__)

SNA_DIR = Path("./sna")
STAGE = "sna"
SCHEMA_VERSION = "ep24-sna/1.0"

#: Node kinds. Kept to what the issue lists as understandable relationships.
NODE_CONTENT = "content"
NODE_ACCOUNT = "account"
NODE_ACTOR = "actor"
NODE_CONCEPT = "concept"
NODE_THEME = "theme"

#: Edge relations. ``mentions``/``posted``/``associated_with`` are derived from
#: the record itself; support/opposition and other stance relations only ever come
#: from evidence the pipeline already extracted (LLM edges or discourse output),
#: never from a guess here.
REL_MENTIONS = "mentions"
REL_POSTED = "posted"
REL_ASSOCIATED_WITH = "associated_with"
REL_SUPPORTS = "supports"
REL_OPPOSES = "opposes"
REL_EXTRACTED = "extracted_relation"

EVIDENCE_DERIVED = "record_derived"
EVIDENCE_EXTRACTED = "pipeline_extracted"

NODES_HEADER = (
    "id", "node_type", "label", "country", "platform",
    "source_id", "source_url", "timestamp", "provenance", "degree",
    "in_degree", "out_degree", "weighted_degree",
)
EDGES_HEADER = (
    "id", "source", "target", "relation", "weight", "directed",
    "evidence", "evidence_quote", "country", "platform", "source_id", "provenance",
)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

def _digest(*parts: str) -> str:
    seed = "|".join(str(p or "").strip() for p in parts)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def content_id(row: Mapping[str, Any], *, country: str = "", index: int = 0) -> str:
    """Canonical identity of one analysed item.

    Mirrors the repository rule in ``roihu_rdf.document_node``: the canonical
    ``video_id`` wins, and the row index is only a fallback for legacy rows that
    carry no identifier at all. Never random, so re-running produces the same ids.
    """
    video_id = str(ep24_value(row, "video_id") or "").strip()
    if video_id:
        return f"content:{_digest(country, video_id)}"
    fallback = str(ep24_value(row, "allas_filename") or "").strip()
    if fallback:
        return f"content:{_digest(country, fallback)}"
    return f"content:{_digest(country, 'row', str(index))}"


def account_id(value: str, *, country: str = "") -> str:
    return f"account:{_digest(country, fold_key(value))}"


def actor_id(value: str, *, country: str = "") -> str:
    """Actor identity, folded so surface variants share one node.

    Note this is the *SNA* node identity, not an entity-registry id. A canonical
    ``entity_id`` from ``ep24_entities`` is preferred by the caller when present.
    """
    return f"actor:{_digest(country, fold_key(value))}"


def concept_id(value: str) -> str:
    return f"concept:{_digest(fold_key(value))}"


def theme_id(value: str) -> str:
    return f"theme:{_digest(fold_key(value))}"


def edge_id(source: str, relation: str, target: str) -> str:
    return f"edge:{_digest(source, relation, target)}"


# --------------------------------------------------------------------------- #
# Graph model
# --------------------------------------------------------------------------- #

@dataclass
class NetworkNode:
    id: str
    node_type: str
    label: str
    country: str = ""
    platform: str = ""
    source_id: str = ""
    source_url: str = ""
    timestamp: str = ""
    provenance: str = ""
    # filled by metrics
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    weighted_degree: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "node_type": self.node_type, "label": self.label,
            "country": self.country, "platform": self.platform,
            "source_id": self.source_id, "source_url": self.source_url,
            "timestamp": self.timestamp, "provenance": self.provenance,
            "degree": self.degree, "in_degree": self.in_degree,
            "out_degree": self.out_degree, "weighted_degree": f"{self.weighted_degree:g}",
        }


@dataclass
class NetworkEdge:
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    directed: bool = True
    evidence: str = ""
    evidence_quote: str = ""
    country: str = ""
    platform: str = ""
    source_id: str = ""
    provenance: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "target": self.target,
            "relation": self.relation, "weight": f"{self.weight:g}",
            "directed": "true" if self.directed else "false",
            "evidence": self.evidence, "evidence_quote": self.evidence_quote,
            "country": self.country, "platform": self.platform,
            "source_id": self.source_id, "provenance": self.provenance,
        }


@dataclass
class Network:
    """Nodes and edges, deduplicated by id. Insertion order is preserved."""

    nodes: dict[str, NetworkNode] = field(default_factory=dict)
    edges: dict[str, NetworkEdge] = field(default_factory=dict)

    def add_node(self, node: NetworkNode) -> NetworkNode:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        # Keep the first label but fill in blanks from later observations, so a
        # node first seen via a sparse row still gains its URL/platform later.
        for attr in ("country", "platform", "source_url", "timestamp", "provenance", "label"):
            if not getattr(existing, attr) and getattr(node, attr):
                setattr(existing, attr, getattr(node, attr))
        return existing

    def add_edge(self, edge: NetworkEdge) -> NetworkEdge:
        """Add an edge, merging duplicates by (source, relation, target).

        A repeated relation between the same pair in the same direction is the
        same tie observed more than once, so its weight accumulates rather than
        producing parallel edges.
        """
        existing = self.edges.get(edge.id)
        if existing is None:
            self.edges[edge.id] = edge
            return edge
        existing.weight += edge.weight
        if not existing.evidence_quote and edge.evidence_quote:
            existing.evidence_quote = edge.evidence_quote
        return existing

    def __len__(self) -> int:
        return len(self.nodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [n.as_row() for n in self.nodes.values()],
            "edges": [e.as_row() for e in self.edges.values()],
        }


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def _cell(row: Mapping[str, Any], column: str) -> str:
    """Read a plain cell without the legacy-alias behaviour of ``ep24_value``.

    ``_split_values`` is used on list columns whose names are not in
    ``LEGACY_ALIASES``; going through ``ep24_value`` here would be a category
    error (it takes a *row*, not a value).
    """
    raw = row.get(column, "")
    text = "" if raw is None else str(raw).strip()
    return "" if text.lower() == "nan" else text


def _registry_positions(row: Mapping[str, Any]) -> dict[str, str]:
    """Map a *mention* to the canonical entity id it resolved to.

    This is the fix for a real fragmentation bug found while building this module:
    the first version paired ``ep24_entity_ids`` with mentions **by list position**,
    but the two lists are not positionally aligned — ``ep24_entity_ids`` holds the
    *deduplicated* ids while the mention list holds surface forms including
    title-stripped variants. One actor then became three nodes ("Petteri Orpo" via
    the registry id, plus "Orpo" and "Petteri Orpo" via the fallback).

    The correct pairing is by *resolution record*: the appended
    ``ep24_entity_resolution_json`` (written by ``ep24_entities.resolve_dataframe``)
    carries, per mention, both the surface form and the canonical name/entity id.
    That is authoritative, so it is preferred; the flat columns are only a fallback.
    """
    mapping: dict[str, str] = {}
    records = _json_list(row.get("ep24_entity_resolution_json"))
    for record in records:
        if not isinstance(record, Mapping):
            continue
        entity_id = str(record.get("entity_id") or "").strip()
        if not entity_id:
            continue
        canonical = str(record.get("canonical_name") or "").strip()
        for form in (
            record.get("surface_form"),
            record.get("normalized_form"),
            canonical,
        ):
            key = _actor_key(str(form or ""))
            if key:
                mapping.setdefault(key, entity_id)
        for variant in record.get("variants") or []:
            key = _actor_key(str(variant or ""))
            if key:
                mapping.setdefault(key, entity_id)

    # Deliberately NO alignment of the flat `ep24_entity_ids` /
    # `ep24_entity_canonical_names` columns. Both are written *sorted" but
    # independently, so zipping them is a guess and produced a real mispairing
    # ("Petteri Orpo" labelled with another person's id). Identity by canonical
    # *name* is safe and is handled in ``resolve_actor_id``; identity by canonical
    # *id* requires the per-mention resolution records above.
    return mapping


def _canonical_name_keys(row: Mapping[str, Any]) -> set[str]:
    """Folded canonical entity names known for this row.

    Used to group a mention under its canonical entity *name* when no per-mention
    resolution record ties it to a registry id. Grouping by name cannot mis-attribute
    an id, and it is deterministic.
    """
    keys: set[str] = set()
    for record in _json_list(row.get("ep24_entity_resolution_json")):
        if isinstance(record, Mapping):
            key = _actor_key(str(record.get("canonical_name") or ""))
            if key:
                keys.add(key)
    for name in _json_list(row.get("ep24_entity_canonical_names")):
        key = _actor_key(str(name or ""))
        if key:
            keys.add(key)
    return keys


def _actor_key(value: str) -> str:
    """Fold an actor mention so a title-stripped form matches its full form."""
    text = _actor_label(value)
    return fold_key(text) if text else ""


def resolve_actor_id(
    actor: str,
    *,
    row: Mapping[str, Any],
    country: str = "",
    registry_positions: Mapping[str, str] | None = None,
    fallback_position: int = -1,
) -> str:
    """Stable node id for an actor mention.

    Preference order, all deterministic and reproducible across runs:

    1. the canonical registry entity id, when the entity-normalization layer
       resolved this mention (checked via the *record*, not by list position);
    2. a folded surface-form id otherwise, so two spellings of one name still land
       on one node.

    There is deliberately no positional fallback into ``ep24_entity_ids``: guessing
    an alignment between two differently-shaped lists is what fragmented actors.
    """
    del fallback_position  # kept in the signature for callers; alignment by index is unsafe
    mapping = registry_positions if registry_positions is not None else _registry_positions(row)
    key = _actor_key(actor)
    if key:
        canonical = mapping.get(key)
        if canonical:
            return f"entity:{canonical}"
        # Group by canonical *name* when a per-mention registry id is unavailable.
        # This cannot mis-attribute an id and keeps the title-stripped and full
        # forms of one actor on a single node. A bare surname ("Orpo") is grouped
        # under the one canonical name it is a part of ("Petteri Orpo") -- only
        # when exactly one canonical name matches, so two Orpos never collapse.
        canonical_names = _canonical_name_keys(row)
        if key in canonical_names:
            return f"entity:{_digest(key)}"
        parts = {name for name in canonical_names if key in name.split()}
        if len(parts) == 1:
            return f"entity:{_digest(next(iter(parts)))}"
        # A title-stripped mention ("Pääministeri Orpo" -> "Orpo") is the same
        # actor as a full form present in this row, so prefer the full form's id
        # rather than creating a second node for it.
        full_forms = [f for f in mapping if f != key and (f.endswith(key) or f.startswith(key))]
        if len(full_forms) == 1:
            return f"entity:{mapping[full_forms[0]]}"
    return actor_id(actor, country=country)


def _split_values(row: Mapping[str, Any], column: str) -> list[str]:
    """Split a legacy list column (comma-separated, as postprocess writes it)."""
    text = _cell(row, column)
    if not text or text in {"[]", "{}"}:
        return []
    raw = text.replace("\r", "\n").replace(";", ",").replace("\n", ",")
    out: list[str] = []
    for item in raw.split(","):
        cleaned = item.strip().strip("\"'").strip().strip("[]").strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _actor_label(value: str) -> str:
    """Label an actor mention, dropping a leading title for display only."""
    stripped, _titles = strip_titles(str(value or "").strip())
    return (stripped or str(value or "").strip()).strip()


def build_network(
    rows: Iterable[Mapping[str, Any]],
    *,
    country: str = "",
    language: str = "",
    include_extracted: bool = True,
) -> Network:
    """Build the evidence-supported graph from accumulated pipeline rows.

    Only relationships the record itself carries are derived:
      * account  -> posted          -> content
      * content  -> mentions        -> actor      (entities / normalized entities)
      * content  -> associated_with -> theme
      * actor    -> associated_with -> concept    (only when the row ties them)

    ``support`` / ``oppose`` relations are never inferred here. They arrive only
    from an already-extracted pipeline edge (``sna_edges_json``), because a stance
    is exactly the kind of claim this layer must not fabricate.
    """
    network = Network()
    platform_default = ""

    for index, row in enumerate(rows):
        row_country = str(ep24_value(row, "country") or country).strip()
        platform = str(ep24_value(row, "source_type") or platform_default).strip()
        cid = content_id(row, country=row_country or country, index=index)
        url = str(ep24_value(row, "allas_filename") or "").strip()
        timestamp = str(
            ep24_value(row, "corrected_date")
            or ep24_value(row, "source_recording")
            or ""
        ).strip()

        content_node = network.add_node(NetworkNode(
            id=cid, node_type=NODE_CONTENT,
            label=str(ep24_value(row, "video_id") or f"row-{index}").strip(),
            country=row_country, platform=platform, source_id=cid,
            source_url=url, timestamp=timestamp, provenance=STAGE,
        ))
        if not content_node.id:  # pragma: no cover - defensive
            continue

        author = str(ep24_value(row, "author_username") or "").strip()
        if author:
            aid = account_id(author, country=row_country)
            network.add_node(NetworkNode(
                id=aid, node_type=NODE_ACCOUNT, label=author,
                country=row_country, platform=platform, source_id=cid,
                source_url=url, provenance=STAGE,
            ))
            network.add_edge(NetworkEdge(
                id=edge_id(aid, REL_POSTED, cid), source=aid, target=cid,
                relation=REL_POSTED, evidence=EVIDENCE_DERIVED,
                country=row_country, platform=platform, source_id=cid,
                provenance=STAGE,
            ))

        # Entities -> actor nodes.
        actors = [*_split_values(row, "entities"), *_split_values(row, "new_entity")]
        registry = _registry_positions(row)
        for raw_actor in actors:
            actor = _actor_label(raw_actor)
            if not actor:
                continue
            aid = resolve_actor_id(
                actor, row=row, country=row_country, registry_positions=registry,
            )
            network.add_node(NetworkNode(
                id=aid, node_type=NODE_ACTOR, label=actor,
                country=row_country, platform=platform, source_id=cid,
                source_url=url, provenance=STAGE,
            ))
            network.add_edge(NetworkEdge(
                id=edge_id(cid, REL_MENTIONS, aid), source=cid, target=aid,
                relation=REL_MENTIONS, evidence=EVIDENCE_DERIVED,
                country=row_country, platform=platform, source_id=cid,
                provenance=STAGE,
            ))

        for theme in _split_values(row, "themes"):
            tid = theme_id(theme)
            network.add_node(NetworkNode(
                id=tid, node_type=NODE_THEME, label=theme,
                country=row_country, platform=platform, source_id=cid,
                provenance=STAGE,
            ))
            network.add_edge(NetworkEdge(
                id=edge_id(cid, REL_ASSOCIATED_WITH, tid), source=cid, target=tid,
                relation=REL_ASSOCIATED_WITH, evidence=EVIDENCE_DERIVED,
                country=row_country, platform=platform, source_id=cid,
                provenance=STAGE,
            ))

        if include_extracted:
            for extracted in _json_list(row.get("sna_edges_json")):
                if not isinstance(extracted, Mapping):
                    continue
                source_label = _actor_label(str(extracted.get("source_actor") or ""))
                target_label = _actor_label(str(extracted.get("target_actor") or ""))
                relation = str(extracted.get("relation_type") or REL_EXTRACTED).strip()
                if not source_label or not target_label:
                    continue
                sid = resolve_actor_id(
                    source_label, row=row, country=row_country,
                    registry_positions=registry,
                )
                tid = resolve_actor_id(
                    target_label, row=row, country=row_country,
                    registry_positions=registry,
                )
                for node_id, label in ((sid, source_label), (tid, target_label)):
                    network.add_node(NetworkNode(
                        id=node_id, node_type=NODE_ACTOR, label=label,
                        country=row_country, platform=platform, source_id=cid,
                        provenance=f"{STAGE}:extracted",
                    ))
                network.add_edge(NetworkEdge(
                    id=edge_id(sid, relation, tid), source=sid, target=tid,
                    relation=relation, directed=bool(extracted.get("directed", True)),
                    evidence=EVIDENCE_EXTRACTED,
                    evidence_quote=str(extracted.get("evidence_quote") or "").strip(),
                    country=row_country, platform=platform, source_id=cid,
                    provenance=f"{STAGE}:extracted",
                ))
    compute_metrics(network)
    reconcile_actor_nodes(network)
    return network


def reconcile_actor_nodes(network: Network) -> Network:
    """Merge actor nodes that represent the same person across rows.

    A row that carries entity-resolution records yields a canonical
    ``entity:<id>`` node; a row that does not (or a mention the registry did not
    resolve) yields a folded-form ``actor:<hash>`` node. Both describe the same
    person, so leaving them separate fragments one actor into several nodes -- the
    exact failure this layer exists to prevent, and one that only appears when
    different rows carry different amounts of entity output.

    Rows are reconciled on the folded canonical *name*, and only when the mapping
    is unambiguous: an ``actor:`` node is merged into an ``entity:`` node when
    exactly one canonical name matches it. Two different people who share a
    surname are never merged, because their full names do not match.
    """
    rename: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for node in network.nodes.values():
        if node.node_type != NODE_ACTOR or not node.id.startswith("entity:"):
            continue
        for key in {_actor_key(node.label), fold_key(node.label)}:
            if key:
                by_label.setdefault(key, node.id)

    for node in network.nodes.values():
        if node.node_type != NODE_ACTOR or node.id.startswith("entity:"):
            continue
        key = _actor_key(node.label)
        candidates: set[str] = set()
        if key in by_label:
            candidates.add(by_label[key])
        else:
            # A bare surname reaches the canonical node only when exactly one
            # canonical actor contains it as a token.
            contains = {
                canonical for label, canonical in by_label.items()
                if key and key in label.split()
            }
            candidates |= contains
        if len(candidates) == 1:
            rename[node.id] = next(iter(candidates))

    if not rename:
        return network

    for old_id, new_id in rename.items():
        node = network.nodes.pop(old_id, None)
        target = network.nodes.get(new_id)
        if node is not None and target is not None:
            for attr in ("country", "platform", "source_url", "timestamp"):
                if not getattr(target, attr) and getattr(node, attr):
                    setattr(target, attr, getattr(node, attr))
        LOG.debug("reconciled actor node %s -> %s", old_id, new_id)

    merged: dict[str, NetworkEdge] = {}
    for edge in network.edges.values():
        source = rename.get(edge.source, edge.source)
        target = rename.get(edge.target, edge.target)
        if edge.source in rename:
            edge.source = source
        if edge.target in rename:
            edge.target = target
        if source == target:
            # A mention of X by X's own post collapses to a self-loop, which
            # carries no network information; drop it rather than inventing one.
            continue
        edge.id = edge_id(source, edge.relation, target)
        existing = merged.get(edge.id)
        if existing is None:
            merged[edge.id] = edge
        else:
            existing.weight += edge.weight
            if not existing.evidence_quote and edge.evidence_quote:
                existing.evidence_quote = edge.evidence_quote
    network.edges = merged
    compute_metrics(network)
    return network


def _json_list(value: Any) -> list[Any]:
    """Parse a JSON list cell, tolerating empty/None/legacy encodings."""
    text = str(value or "").strip()
    if not text or text in {"[]", "{}"}:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    return list(parsed) if isinstance(parsed, list) else []


# --------------------------------------------------------------------------- #
# Metrics -- basic and interpretable only
# --------------------------------------------------------------------------- #

def compute_metrics(network: Network) -> Network:
    """Degree-family metrics. No community detection, no centrality zoo."""
    for node in network.nodes.values():
        node.degree = node.in_degree = node.out_degree = 0
        node.weighted_degree = 0.0
    for edge in network.edges.values():
        for node_id, direction in ((edge.source, "out"), (edge.target, "in")):
            node = network.nodes.get(node_id)
            if node is None:
                continue
            if direction == "out":
                node.out_degree += 1
            else:
                node.in_degree += 1
            node.weighted_degree += float(edge.weight)
    for node in network.nodes.values():
        node.degree = node.in_degree + node.out_degree
    return network


def connected_components(network: Network) -> list[list[str]]:
    """Weakly-connected components (direction ignored), largest first."""
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in network.nodes}
    for edge in network.edges.values():
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in adjacency.get(current, ()):  # type: ignore[arg-type]
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(group))
    components.sort(key=len, reverse=True)
    return components


def density(network: Network) -> float:
    """Edge density for a simple directed graph, in ``[0, 1]``."""
    n = len(network.nodes)
    if n < 2:
        return 0.0
    possible = n * (n - 1)
    return len(network.edges) / possible if possible else 0.0


def graph_summary(network: Network) -> dict[str, Any]:
    """The structured summary fed to the interpretation layer and the report."""
    node_types: dict[str, int] = {}
    for node in network.nodes.values():
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
    relations: dict[str, int] = {}
    for edge in network.edges.values():
        relations[edge.relation] = relations.get(edge.relation, 0) + 1
    components = connected_components(network)
    top = sorted(
        network.nodes.values(), key=lambda n: (-n.degree, -n.weighted_degree, n.label)
    )[:10]
    countries = sorted({n.country for n in network.nodes.values() if n.country})
    platforms = sorted({n.platform for n in network.nodes.values() if n.platform})
    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": len(network.nodes),
        "edges": len(network.edges),
        "node_types": node_types,
        "relations": relations,
        "density": round(density(network), 6),
        "components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "countries": countries,
        "platforms": platforms,
        "top_nodes": [
            {
                "id": n.id, "label": n.label, "node_type": n.node_type,
                "degree": n.degree, "in_degree": n.in_degree,
                "out_degree": n.out_degree, "weighted_degree": round(n.weighted_degree, 4),
            }
            for n in top
        ],
    }


# --------------------------------------------------------------------------- #
# Castells interpretation
# --------------------------------------------------------------------------- #

CASTELLS_SYSTEM_PROMPT = """You interpret an already-computed social network graph through \
Manuel Castells' theory of communication power and the network society.

Absolute rules:
- You did NOT build the graph. Never add, remove or restate a node or an edge that \
is not in the supplied summary.
- Never infer a strategic role (programmer, switcher, broker) unless the summary's \
structure actually demonstrates it. With small graphs, say the evidence is insufficient.
- Separate observation from interpretation. Label interpretation as interpretation.
- Do not overclaim causality, influence or coordination.

Write about: networks of communication; inclusion in and exclusion from networks; \
nodes and flows; communication power; network-making power; and the relationships \
between political actors, accounts, concepts and discourse.

Be concise. Prefer "the graph shows X, which under Castells can be read as Y" over \
assertions of fact about power."""


def castells_interpretation(
    summary: Mapping[str, Any],
    *,
    client: Any = None,
    model: str = "",
) -> str:
    """Interpret a graph summary through Castells. Returns Markdown.

    ``client`` is an injected callable ``client(prompt, system) -> str`` (or an
    object with a matching ``chat``); tests pass a stub so no model is needed. With
    no client, a deterministic structural reading is produced instead of calling
    out — the report never has an empty interpretation section, and nothing is
    fabricated because the fallback only restates measured structure.
    """
    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)
    prompt = (
        "Here is the computed graph summary as JSON. Interpret it.\n\n"
        f"```json\n{summary_json}\n```"
    )
    if client is None:
        return _structural_reading(summary)
    try:
        if callable(client):
            text = client(prompt, CASTELLS_SYSTEM_PROMPT)
        else:  # duck-typed ollama-style client
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": CASTELLS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0},
            )
            text = response["message"]["content"]
        cleaned = str(text or "").strip()
        return cleaned or _structural_reading(summary)
    except Exception:  # noqa: BLE001 - interpretation must never fail the stage
        LOG.exception("Castells interpretation failed; falling back to structural reading")
        return _structural_reading(summary)


def _structural_reading(summary: Mapping[str, Any]) -> str:
    """Deterministic, non-speculative reading of the measured graph.

    Only restates counts and the top-degree nodes, framed as a network reading.
    This is what makes the fallback safe: it cannot introduce a claim that the
    graph does not support.
    """
    nodes = summary.get("nodes", 0)
    edges = summary.get("edges", 0)
    relations = summary.get("relations") or {}
    top = summary.get("top_nodes") or []
    components = summary.get("components", 0)
    lines = [
        "*Interpretation (deterministic structural reading; no model was consulted).*",
        "",
        f"The graph contains **{nodes} node(s)** and **{edges} edge(s)** across "
        f"**{components} weakly-connected component(s)**, with density "
        f"**{summary.get('density', 0.0)}**.",
        "",
    ]
    if relations:
        listed = ", ".join(f"`{name}` ({count})" for name, count in sorted(relations.items()))
        lines.append(f"Observed relations: {listed}.")
        lines.append("")
    if top:
        lines.append("Most connected nodes:")
        for node in top[:5]:
            lines.append(
                f"- **{node.get('label', '')}** ({node.get('node_type', '')}) — "
                f"degree {node.get('degree', 0)}, in {node.get('in_degree', 0)}, "
                f"out {node.get('out_degree', 0)}"
            )
        lines.append("")
    lines.append(
        "Read through Castells, the degree distribution describes which actors, accounts and "
        "concepts are positioned as nodes through which this sample's communication flows, and "
        "which are excluded from the observed network. The counts above are the whole empirical "
        "basis for that reading."
    )
    if nodes < 10 or edges < 10:
        lines.append("")
        lines.append(
            "**Caveat:** the graph is small. Network-making power, programmer and switcher roles "
            "are *not* identifiable from a graph of this size, and are deliberately not claimed."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def render_report(
    network: Network,
    *,
    country: str = "",
    language: str = "",
    run_metadata: Mapping[str, Any] | None = None,
    interpretation: str = "",
) -> str:
    """Human-readable Markdown report for a researcher to read directly."""
    summary = graph_summary(network)
    meta = dict(run_metadata or {})
    lines = [
        "# Social Network Analysis",
        "",
        "## Dataset",
        "",
        f"- Country: {country or 'n/a'}",
        f"- Language: {language or 'n/a'}",
        f"- Schema: {SCHEMA_VERSION}",
        f"- Nodes: {summary['nodes']}",
        f"- Edges: {summary['edges']}",
        "",
        "## Network construction",
        "",
        "Nodes and edges are derived only from relationships the accumulated pipeline record "
        "supports. Content nodes come from the canonical item identity; account nodes from the "
        "posting account; actor nodes from extracted entities and normalized entities; theme "
        "nodes from extracted themes. Stance relations (`supports` / `opposes`) appear only if a "
        "prior stage already extracted them — they are never inferred here.",
        "",
        "## Nodes",
        "",
    ]
    by_type: dict[str, int] = {}
    for node in network.nodes.values():
        by_type[node.node_type] = by_type.get(node.node_type, 0) + 1
    for node_type, count in sorted(by_type.items()):
        lines.append(f"- `{node_type}`: {count}")
    if not by_type:
        lines.append("- (none)")

    lines += ["", "## Edges", ""]
    if summary["relations"]:
        for relation, count in sorted(summary["relations"].items()):
            lines.append(f"- `{relation}`: {count}")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Basic network statistics",
        "",
        f"- Density: {summary['density']}",
        f"- Weakly-connected components: {summary['components']}",
        f"- Largest component size: {summary['largest_component_size']}",
        f"- Countries: {', '.join(summary['countries']) or 'n/a'}",
        f"- Platforms: {', '.join(summary['platforms']) or 'n/a'}",
        "",
        "## Most connected nodes",
        "",
    ]
    if summary["top_nodes"]:
        lines += [
            "| node | type | degree | in | out | weighted |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for node in summary["top_nodes"]:
            lines.append(
                f"| {node['label']} | `{node['node_type']}` | {node['degree']} | "
                f"{node['in_degree']} | {node['out_degree']} | {node['weighted_degree']} |"
            )
    else:
        lines.append("(no nodes)")

    lines += [
        "",
        "## Main observed relationships",
        "",
    ]
    if network.edges:
        examples = list(network.edges.values())[:15]
        for edge in examples:
            source = network.nodes.get(edge.source)
            target = network.nodes.get(edge.target)
            lines.append(
                f"- {source.label if source else edge.source} —`{edge.relation}`→ "
                f"{target.label if target else edge.target}"
                f" ({edge.evidence})"
            )
        if len(network.edges) > len(examples):
            lines.append(f"- … and {len(network.edges) - len(examples)} more edge(s).")
    else:
        lines.append("(no edges)")

    lines += [
        "",
        "## Castellsian interpretation",
        "",
        "*The section below is theory-informed interpretation, not measurement. The graph itself "
        "is built upstream without reference to any theory; this lens is applied only to the "
        "computed summary above.*",
        "",
        interpretation or _structural_reading(summary),
        "",
        "## Caveats and uncertainty",
        "",
        "- The graph reflects only what the accumulated record supports. Absence of an edge is "
        "not evidence that no relationship exists in the world, only that this sample does not "
        "show one.",
        "- Stance relations depend on prior extraction stages; where those are absent, only "
        "mention, posting and theme associations appear.",
        "- Network-making power and programmer/switcher roles are not identifiable from small "
        "graphs and are not claimed.",
        "",
        "## Provenance / run metadata",
        "",
    ]
    for key, value in sorted(meta.items()):
        lines.append(f"- {key}: {value}")
    lines.append(f"- stage: {STAGE}")
    lines.append(f"- schema_version: {SCHEMA_VERSION}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    network: Network,
    *,
    language: str = "",
    country: str = "",
    output_dir: str | Path = SNA_DIR,
    run_metadata: Mapping[str, Any] | None = None,
    interpretation: str = "",
) -> dict[str, str]:
    """Write the node/edge tables, JSON and Markdown report. Returns the paths.

    The CSV names and the ``id`` / ``label`` / ``source`` / ``target`` columns are
    the contract ``roihu_rdf.maybe_emit_network`` reads, so step 9 consumes these
    tables directly instead of translating a second schema.
    """
    out = Path(output_dir)
    suffix = language or "all"
    paths = {
        "nodes_csv": out / f"sna_nodes_{suffix}.csv",
        "edges_csv": out / f"sna_edges_{suffix}.csv",
        "network_json": out / f"sna_network_{suffix}.json",
        "report_md": out / f"sna_report_{suffix}.md",
    }
    _write_csv(paths["nodes_csv"], NODES_HEADER, (n.as_row() for n in network.nodes.values()))
    _write_csv(paths["edges_csv"], EDGES_HEADER, (e.as_row() for e in network.edges.values()))
    payload = {
        **network.as_dict(),
        "summary": graph_summary(network),
        "country": country,
        "language": language,
        "run_metadata": dict(run_metadata or {}),
    }
    paths["network_json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["report_md"].write_text(
        render_report(
            network, country=country, language=language,
            run_metadata=run_metadata, interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def append_to_dataframe(frame: Any, network: Network, *, interpretation: str = "") -> Any:
    """Append SNA fields per row, preserving every existing column.

    Additive by the repository contract. The per-row node/edge lists are filtered
    by ``source_id`` so a row carries the part of the graph it contributed, while
    the aggregate tables live in the ``sna/`` files.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("append_to_dataframe expects a pandas DataFrame")
    per_row: dict[str, dict[str, list[str]]] = {}
    for node in network.nodes.values():
        bucket = per_row.setdefault(node.source_id, {"nodes": [], "edges": []})
        bucket["nodes"].append(node.id)
    for edge in network.edges.values():
        bucket = per_row.setdefault(edge.source_id, {"nodes": [], "edges": []})
        bucket["edges"].append(edge.id)

    for column in ("sna_node_ids", "sna_edge_ids", "sna_summary_json", "sna_report_markdown"):
        if column not in frame.columns:
            frame[column] = ""
    summary = json.dumps(graph_summary(network), ensure_ascii=False, sort_keys=True)
    for index, row in frame.iterrows():
        cid = content_id(row, country=str(ep24_value(row, "country") or ""), index=index)
        bucket = per_row.get(cid, {"nodes": [], "edges": []})
        frame.at[index, "sna_node_ids"] = json.dumps(bucket["nodes"], ensure_ascii=False)
        frame.at[index, "sna_edge_ids"] = json.dumps(bucket["edges"], ensure_ascii=False)
        frame.at[index, "sna_summary_json"] = summary
        if interpretation:
            frame.at[index, "sna_report_markdown"] = interpretation
    return frame
