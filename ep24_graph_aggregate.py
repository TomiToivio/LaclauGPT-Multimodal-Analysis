"""Read-only aggregation of EP24 graph layers across videos, countries and steps (#380).

Step 8 persists one *event layer* per video: a multigraph of typed actor→actor relations, written as
`sna_nodes.csv` / `sna_edges.csv` beside each country export. Step 7 persists DNA statements, one
actor making one claim about one concept. Neither is a network: a density computed inside one video
is a fact about that video, and a Step 7 statement is not a social tie.

This module turns those persisted records into **views**, and keeps three things apart that are easy
to blur:

* **layers** — Step 7 coding (actor–concept), Step 8 observed communication/affiliation
  (actor–actor). They are never summed into one weighted edge list;
* **events and ties** — events are deduplicated by their own id, then repeated ties are aggregated
  with a declared grouping key, so a weight always carries the item ids that produced it;
* **a multigraph and its projection** — parallel typed relations are kept (two actors can both
  mention and quote each other), and any one-mode metric is computed on the documented simple
  projection, then labelled as such.

The combined view is a heterogeneous multilayer graph. It is deliberately **not** offered as one-mode
SNA: `metrics_for_view` refuses to compute degree/density across layers unless a single layer is
selected, because mixing an actor–concept coding edge with an actor–actor mention edge produces a
number with no referent.

Everything here reads; nothing here writes to the corpus, Mongo or the object store.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from ep24_sna import basic_metrics, node_identity

SCHEMA_VERSION = "ep24.graph_aggregate/1"

LAYER_STEP7 = "step7_dna"
LAYER_STEP8 = "step8_social"
LAYERS = (LAYER_STEP7, LAYER_STEP8)

VIEW_STEP7 = "step7"
VIEW_STEP8 = "step8"
VIEW_COMBINED = "combined"
VIEWS = (VIEW_STEP7, VIEW_STEP8, VIEW_COMBINED)

#: Relation families that describe an observed actor→actor relation in Step 8. Institutional
#: affiliation is kept separate from communication: an organisation membership is not a message.
STEP8_RELATIONS = frozenset(
    {
        "communication_mention",
        "mention",
        "reply",
        "quotation",
        "repost",
        "direct_interaction",
        "institutional_affiliation",
        "affiliation",
        "reported_statement",
    }
)

#: rDNA agreement values that may label an actor–concept coding edge.
DNA_AGREEMENTS = frozenset({"agree", "disagree", "neutral", "mixed"})


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    text = _text(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


# -- loading ---------------------------------------------------------------------------------


def load_step8_edges(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read Step 8 `*.sna_edges.csv` exports and normalise them to the aggregation contract."""
    edges: list[dict[str, Any]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source = _text(row.get("source") or row.get("source_node_id"))
                target = _text(row.get("target") or row.get("target_node_id"))
                relation = _text(row.get("relation_type") or row.get("edge_type"))
                if not source or not target or not relation:
                    continue
                edges.append(
                    {
                        "layer": LAYER_STEP8,
                        "edge_id": _text(row.get("edge_id"))
                        or f"step8:{source}|{relation}|{target}|{_text(row.get('canonical_item_id'))}",
                        "source": source,
                        "target": target,
                        "source_label": _text(row.get("source_label")) or source,
                        "target_label": _text(row.get("target_label")) or target,
                        "source_type": "actor",
                        "target_type": "actor",
                        "relation": relation,
                        "directed": _text(row.get("directed")).lower() not in {"false", "0", "no"},
                        "weight": 1.0,
                        "country": _text(row.get("country")),
                        "platform": _text(row.get("platform")),
                        "item_id": _text(row.get("canonical_item_id")),
                        "timestamp": _text(row.get("timestamp")),
                        "evidence_quote": _text(row.get("evidence_quote")),
                        "evidence_modality": _text(row.get("evidence_modality")),
                        "confidence": _float(row.get("confidence") or row.get("model_confidence")),
                        "network_layer": _text(row.get("network_layer")) or "communication",
                    }
                )
    return edges


def dna_edges(statements: Iterable[Mapping[str, Any]], *, item_id: str = "") -> list[dict[str, Any]]:
    """One actor–concept coding edge per DNA statement (Step 7). Never an actor→actor tie.

    A statement asserts that *this actor* made *this claim* about *this concept*. Two statements by
    two actors about the same concept are two codings of the same topic, which is not a relation
    between the actors — the reason this layer is kept out of Step 8 arithmetic.
    """
    edges: list[dict[str, Any]] = []
    for statement in statements:
        actor = _text(statement.get("organization") or statement.get("actor"))
        concept = _text(statement.get("concept"))
        if not actor or not concept:
            continue
        actor_id = _text(statement.get("actor_id"))
        concept_id = _text(statement.get("concept_id"))
        actor_node = node_identity(actor, entity_id=actor_id, node_type="actor")
        concept_node = node_identity(concept, entity_id=concept_id, node_type="concept")
        agreement = _text(statement.get("agreement") or statement.get("stance")).casefold()
        relation = agreement if agreement in DNA_AGREEMENTS else "codes"
        statement_id = _text(statement.get("statement_id")) or (
            f"{item_id}:{actor_node}:{concept_node}"
        )
        edges.append(
            {
                "layer": LAYER_STEP7,
                "edge_id": f"step7:{statement_id}",
                "source": actor_node,
                "target": concept_node,
                "source_label": actor,
                "target_label": concept,
                "source_type": "actor",
                "target_type": "concept",
                "relation": relation,
                "directed": True,
                "weight": 1.0,
                "country": _text(statement.get("country")),
                "platform": _text(statement.get("platform")),
                "item_id": item_id or _text(statement.get("document_id")),
                "timestamp": _text(statement.get("date_time")),
                "evidence_quote": _text(statement.get("evidence_quote")),
                "evidence_modality": "dna_statement",
                "confidence": _float(statement.get("confidence")),
                "network_layer": "coding",
            }
        )
    return edges


def dna_edges_from_cell(raw: Any, *, item_id: str = "") -> list[dict[str, Any]]:
    """Convenience for a `dna_statements_json` cell."""
    return dna_edges(_json_list(raw), item_id=item_id)


# -- events, aggregation, filtering ----------------------------------------------------------


def dedupe_events(edges: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per event id. A re-read export must not double a tie's weight."""
    seen: dict[str, dict[str, Any]] = {}
    for edge in edges:
        event_id = _text(edge.get("edge_id"))
        if not event_id:
            event_id = (
                f"{edge.get('layer')}:{edge.get('source')}|{edge.get('relation')}|"
                f"{edge.get('target')}|{edge.get('item_id')}"
            )
        if event_id in seen:
            seen[event_id].setdefault("duplicate_event_ids", []).append(event_id)
            continue
        seen[event_id] = dict(edge)
    return list(seen.values())


@dataclass
class AggregateKey:
    """What makes two events the same tie: endpoints and relation family, and nothing else."""

    source: str
    target: str
    relation: str


def aggregate_ties(
    edges: Iterable[Mapping[str, Any]],
    *,
    by: Sequence[str] = ("source", "target", "relation", "layer"),
) -> list[dict[str, Any]]:
    """Aggregate events into weighted ties, keeping the provenance that produced the weight.

    Relation families are never merged: three mentions and one quotation between the same pair are
    two ties, because summing them would report a number that no relation in the source supports.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        key = tuple(_text(edge.get(name)) for name in by)
        if not key[0] or not key[1]:
            continue
        groups[key].append(dict(edge))

    ties: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        stamps = sorted(_text(row.get("timestamp")) for row in rows if _text(row.get("timestamp")))
        ties.append(
            {
                "layer": _text(rows[0].get("layer")),
                "source": _text(rows[0].get("source")),
                "target": _text(rows[0].get("target")),
                "source_label": _text(rows[0].get("source_label")),
                "target_label": _text(rows[0].get("target_label")),
                "relation": _text(rows[0].get("relation")),
                "directed": bool(rows[0].get("directed", True)),
                "weight": float(len(rows)),
                "event_count": len(rows),
                "event_ids": sorted({_text(r.get("edge_id")) for r in rows}),
                "item_ids": sorted({_text(r.get("item_id")) for r in rows if _text(r.get("item_id"))}),
                "countries": sorted(
                    {_text(r.get("country")) for r in rows if _text(r.get("country"))}
                ),
                "platforms": sorted(
                    {_text(r.get("platform")) for r in rows if _text(r.get("platform"))}
                ),
                "first_seen": stamps[0] if stamps else "",
                "last_seen": stamps[-1] if stamps else "",
                "countries_per_event": dict(
                    sorted(
                        {
                            country: sum(1 for r in rows if _text(r.get("country")) == country)
                            for country in {_text(r.get("country")) for r in rows}
                            if country
                        }.items()
                    )
                ),
            }
        )
    return ties


def _dimension_values(edge: Mapping[str, Any], plural: str, singular: str) -> set[str]:
    """A tie carries a list, an event carries one value; both answer the same filter question."""
    listed = edge.get(plural)
    if listed:
        return {_text(value) for value in listed if _text(value)}
    single = _text(edge.get(singular))
    return {single} if single else set()


def apply_filters(
    edges: Iterable[Mapping[str, Any]],
    *,
    layers: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    relations: Iterable[str] | None = None,
    actors: Iterable[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    min_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Filter events or ties. Every dimension is optional; an empty filter keeps everything."""
    layer_set = {_text(v) for v in layers or []}
    country_set = {_text(v) for v in countries or []}
    platform_set = {_text(v) for v in platforms or []}
    relation_set = {_text(v) for v in relations or []}
    actor_set = {_text(v) for v in actors or []}

    kept: list[dict[str, Any]] = []
    for edge in edges:
        if layer_set and _text(edge.get("layer")) not in layer_set:
            continue
        # An aggregated tie carries `countries`/`platforms`; a single event carries one of each.
        if country_set and not (country_set & _dimension_values(edge, "countries", "country")):
            continue
        if platform_set and not (platform_set & _dimension_values(edge, "platforms", "platform")):
            continue
        if relation_set and _text(edge.get("relation")) not in relation_set:
            continue
        if actor_set and not actor_set & {_text(edge.get("source")), _text(edge.get("target"))}:
            continue
        stamp = _text(edge.get("timestamp") or edge.get("last_seen"))
        if since and stamp and stamp < since:
            continue
        if until and stamp and stamp > until:
            continue
        if min_weight and _float(edge.get("weight"), 1.0) < min_weight:
            continue
        kept.append(dict(edge))
    return kept


# -- projections and metrics ----------------------------------------------------------------


def simple_projection(
    ties: Iterable[Mapping[str, Any]], *, directed: bool = True
) -> list[dict[str, Any]]:
    """Collapse parallel typed ties into one arc per ordered pair.

    The projection is what density and degree may be computed on, and it must say so: the relation
    families survive as a sorted list, so the reader can still see that the arc is a composite.
    """
    arcs: dict[tuple[str, str], dict[str, Any]] = {}
    for tie in ties:
        source = _text(tie.get("source"))
        target = _text(tie.get("target"))
        if not source or not target or source == target:
            continue
        pair: tuple[str, str] = (
            (source, target) if directed else (min(source, target), max(source, target))
        )
        key = pair
        arc = arcs.setdefault(
            key,
            {
                "source": key[0],
                "target": key[1],
                "weight": 0.0,
                "relations": set(),
                "item_ids": set(),
                "event_count": 0,
            },
        )
        arc["weight"] += _float(tie.get("weight"), 1.0)
        arc["relations"].add(_text(tie.get("relation")))
        arc["item_ids"].update(tie.get("item_ids") or [])
        arc["event_count"] += int(tie.get("event_count") or 0)
    projected: list[dict[str, Any]] = []
    for arc in arcs.values():
        projected.append(
            {
                "source": arc["source"],
                "target": arc["target"],
                "weight": arc["weight"],
                "relations": sorted(arc["relations"]),
                "item_count": len(arc["item_ids"]),
                "event_count": arc["event_count"],
                "directed": directed,
            }
        )
    return sorted(projected, key=lambda row: (row["source"], row["target"]))


def nodes_for(edges: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Node table derived from the edges in a view, with a node type that says what it is."""
    nodes: dict[str, dict[str, Any]] = {}
    for edge in edges:
        layer = _text(edge.get("layer"))
        for role in ("source", "target"):
            node_id = _text(edge.get(role))
            if not node_id:
                continue
            node = nodes.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "label": _text(edge.get(f"{role}_label")) or node_id,
                    "layers": set(),
                    # `node_identity` returns `entity:<id>` for a resolved actor and a hashed
                    # `actor:`/`concept:` id otherwise, so the type travels with the edge rather
                    # than being guessed from the id.
                    "node_type": _text(edge.get(f"{role}_type")) or "actor",
                    "item_ids": set(),
                },
            )
            node["layers"].add(layer)
            node["item_ids"].update(edge.get("item_ids") or [])
    return [
        {
            "node_id": node["node_id"],
            "label": node["label"],
            "node_type": node["node_type"],
            "layers": sorted(node["layers"]),
            "item_count": len(node["item_ids"]),
        }
        for node in nodes.values()
    ]


def metrics_for_view(
    nodes: Iterable[Mapping[str, Any]],
    ties: Iterable[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    """Metrics for one view, or a refusal with a reason.

    A number computed across two layers has no referent: a mention between two actors and an
    agreement between an actor and a concept are not the same unit of observation, so no
    cross-layer degree or density is produced.
    """
    tie_rows = list(ties)
    layers = sorted({_text(tie.get("layer")) for tie in tie_rows if tie.get("layer")})
    if view == VIEW_COMBINED:
        return {
            "schema_version": SCHEMA_VERSION,
            "view": view,
            "metric_status": "not_applicable",
            "layers": layers,
            "reason": (
                "the combined view is a heterogeneous multilayer graph; one-mode SNA metrics are "
                "computed per layer on a declared projection, not across layer boundaries"
            ),
            "layer_metrics": {
                layer: metrics_for_view(
                    nodes_for([tie for tie in tie_rows if _text(tie.get("layer")) == layer]),
                    [tie for tie in tie_rows if _text(tie.get("layer")) == layer],
                    view=layer,
                )
                for layer in layers
            },
        }
    projection = simple_projection(tie_rows, directed=True)
    projection_nodes = nodes_for(tie_rows)
    metrics = basic_metrics(projection_nodes, projection)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "view": view,
            "metric_status": "computed",
            "layers": layers,
            "projection": "simple directed actor pair; parallel typed relations collapsed",
            "relation_families": sorted({_text(tie.get("relation")) for tie in tie_rows}),
            "mean_events_per_tie": (
                mean([_float(tie.get("event_count"), 0) for tie in tie_rows]) if tie_rows else 0.0
            ),
        }
    )
    return metrics


# -- exports --------------------------------------------------------------------------------


def build_view(
    step8_edges: Iterable[Mapping[str, Any]],
    step7_edges: Iterable[Mapping[str, Any]] = (),
    *,
    view: str = VIEW_COMBINED,
    **filters: Any,
) -> dict[str, Any]:
    """Events → deduplicated events → weighted ties → filters → view, nodes and metrics."""
    events = dedupe_events([*step8_edges, *step7_edges])
    if view == VIEW_STEP7:
        events = [edge for edge in events if edge.get("layer") == LAYER_STEP7]
    elif view == VIEW_STEP8:
        events = [edge for edge in events if edge.get("layer") == LAYER_STEP8]
    events = apply_filters(events, **filters)
    ties = aggregate_ties(events)
    return {
        "schema_version": SCHEMA_VERSION,
        "view": view,
        "event_count": len(events),
        "tie_count": len(ties),
        "nodes": nodes_for(ties),
        "ties": ties,
        "edges": simple_projection(ties) if view != VIEW_COMBINED else [],
        "metrics": metrics_for_view(nodes_for(ties), ties, view=view),
        "filters": {key: sorted(value) if isinstance(value, (set, tuple)) else value
                    for key, value in filters.items() if value},
        "heterogeneous": view == VIEW_COMBINED,
        "caveat": (
            "a tie is a count of events in this view's boundary; it is not an intensity, and a "
            "single-video fragment is not the political field"
        ),
    }


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=lambda o: sorted(o)),
        encoding="utf-8",
    )
    return target


def write_edges_csv(ties: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    rows = list(ties)
    columns = [
        "layer",
        "source",
        "source_label",
        "relation",
        "target",
        "target_label",
        "directed",
        "weight",
        "event_count",
        "item_ids",
        "countries",
        "first_seen",
        "last_seen",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "item_ids": ";".join(row.get("item_ids") or []),
                    "countries": ";".join(row.get("countries") or []),
                }
            )
    return target


_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def write_graphml(
    nodes: Iterable[Mapping[str, Any]],
    ties: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """GraphML with the layer, relation and weight retained so the file is not a hairball of names."""
    root = ET.Element(f"{{{_GRAPHML_NS}}}graphml")
    for key_id, key_for, name in (
        ("layer", "edge", "layer"),
        ("relation", "edge", "relation"),
        ("weight", "edge", "weight"),
        ("event_count", "edge", "event_count"),
        ("node_type", "node", "node_type"),
        ("label", "node", "label"),
    ):
        ET.SubElement(
            root,
            f"{{{_GRAPHML_NS}}}key",
            {"id": key_id, "for": key_for, "attr.name": name, "attr.type": "string"},
        )
    graph = ET.SubElement(root, f"{{{_GRAPHML_NS}}}graph", {"edgedefault": "directed"})
    for node in nodes:
        element = ET.SubElement(graph, f"{{{_GRAPHML_NS}}}node", {"id": str(node["node_id"])})
        ET.SubElement(element, f"{{{_GRAPHML_NS}}}data", {"key": "label"}).text = str(
            node.get("label") or node["node_id"]
        )
        ET.SubElement(element, f"{{{_GRAPHML_NS}}}data", {"key": "node_type"}).text = str(
            node.get("node_type") or "actor"
        )
    for index, tie in enumerate(ties):
        element = ET.SubElement(
            graph,
            f"{{{_GRAPHML_NS}}}edge",
            {
                "id": f"tie{index}",
                "source": str(tie["source"]),
                "target": str(tie["target"]),
            },
        )
        for key, value in (
            ("layer", tie.get("layer")),
            ("relation", tie.get("relation")),
            ("weight", tie.get("weight")),
            ("event_count", tie.get("event_count")),
        ):
            ET.SubElement(element, f"{{{_GRAPHML_NS}}}data", {"key": key}).text = str(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def write_gexf(
    nodes: Iterable[Mapping[str, Any]],
    ties: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """GEXF 1.2, the other format a network analyst actually opens."""
    root = ET.Element(
        "gexf",
        {"xmlns": "http://gexf.net/1.2", "version": "1.2"},
    )
    graph = ET.SubElement(root, "graph", {"mode": "static", "defaultedgetype": "directed"})
    node_rows = list(nodes)
    tie_rows = list(ties)
    node_index = {str(node["node_id"]): index for index, node in enumerate(node_rows)}
    node_block = ET.SubElement(graph, "nodes")
    for node in node_rows:
        element = ET.SubElement(
            node_block,
            "node",
            {"id": str(node_index[str(node["node_id"])]), "label": str(node.get("label") or "")},
        )
        attrs = ET.SubElement(element, "attvalues")
        ET.SubElement(
            attrs, "attvalue", {"for": "node_type", "value": str(node.get("node_type") or "actor")}
        )
    edge_block = ET.SubElement(graph, "edges")
    for index, tie in enumerate(tie_rows):
        element = ET.SubElement(
            edge_block,
            "edge",
            {
                "id": f"e{index}",
                "source": str(node_index.get(str(tie["source"]), "")),
                "target": str(node_index.get(str(tie["target"]), "")),
                "weight": str(tie.get("weight")),
                "label": str(tie.get("relation") or ""),
            },
        )
        attrs = ET.SubElement(element, "attvalues")
        ET.SubElement(attrs, "attvalue", {"for": "layer", "value": str(tie.get("layer") or "")})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def write_view_bundle(view: Mapping[str, Any], out_dir: str | Path, *, stem: str | None = None) -> dict[str, str]:
    """Write the four exports a downstream view needs, and return their paths."""
    directory = Path(out_dir)
    name = stem or str(view.get("view") or "combined")
    written = {
        "json": str(write_json(view, directory / f"{name}.json")),
        "edges_csv": str(write_edges_csv(view.get("ties") or [], directory / f"{name}.ties.csv")),
        "graphml": str(
            write_graphml(view.get("nodes") or [], view.get("ties") or [], directory / f"{name}.graphml")
        ),
        "gexf": str(
            write_gexf(view.get("nodes") or [], view.get("ties") or [], directory / f"{name}.gexf")
        ),
    }
    return written


__all__ = [
    "LAYERS",
    "LAYER_STEP7",
    "LAYER_STEP8",
    "SCHEMA_VERSION",
    "STEP8_RELATIONS",
    "VIEWS",
    "VIEW_COMBINED",
    "VIEW_STEP7",
    "VIEW_STEP8",
    "aggregate_ties",
    "apply_filters",
    "build_view",
    "dedupe_events",
    "dna_edges",
    "dna_edges_from_cell",
    "load_step8_edges",
    "metrics_for_view",
    "nodes_for",
    "simple_projection",
    "write_edges_csv",
    "write_gexf",
    "write_graphml",
    "write_json",
    "write_view_bundle",
]

def main(argv: Sequence[str] | None = None) -> int:
    """Build one view from persisted Step 8 exports and DNA statements and write its exports."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ep24-graph-aggregate",
        description="Aggregate EP24 graph layers across videos/countries and export a view (#380).",
    )
    parser.add_argument("--step8-edges", nargs="*", default=[], help="*.sna_edges.csv files")
    parser.add_argument("--dna-json", default=None, help="JSON list of rDNA statements")
    parser.add_argument("--view", choices=VIEWS, default=VIEW_COMBINED)
    parser.add_argument("--country", action="append", default=[])
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--relation", action="append", default=[])
    parser.add_argument("--actor", action="append", default=[])
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    step8 = load_step8_edges(args.step8_edges)
    dna: list[dict[str, Any]] = []
    if args.dna_json:
        dna = dna_edges(json.loads(Path(args.dna_json).read_text(encoding="utf-8")))
    view = build_view(
        step8,
        dna,
        view=args.view,
        countries=args.country,
        platforms=args.platform,
        relations=args.relation,
        actors=args.actor,
        since=args.since,
        until=args.until,
        min_weight=args.min_weight,
    )
    written = write_view_bundle(view, args.out_dir, stem=args.stem)
    print(json.dumps({"view": view["view"], "ties": view["tie_count"], "written": written}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
