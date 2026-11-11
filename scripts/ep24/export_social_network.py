#!/usr/bin/env python3
"""Export a private filtered Step 8 event view from cumulative CSV snapshots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ep24_sna_aggregate import aggregate_social_events


def export_view(view: dict, output: Path) -> None:
    """Create a new private directory, never overwrite an earlier export."""
    previous = os.umask(0o077)
    try:
        _write_view(view, output)
    finally:
        os.umask(previous)


def _write_view(view: dict, output: Path) -> None:
    import networkx as nx

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    (output / "network.json").write_text(json.dumps(view, sort_keys=True, indent=2), encoding="utf-8")
    for table in ("nodes", "events", "projection"):
        rows = view[table]
        fields = sorted({key for row in rows for key in row})
        with (output / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                              for key, value in row.items()} for row in rows)
    graph = nx.MultiDiGraph()
    graph.graph["boundary"] = json.dumps(view["boundary"], sort_keys=True)
    for node in view["nodes"]:
        graph.add_node(node["node_id"], labels=json.dumps(node["labels"]),
                       provenance_item_ids=json.dumps(node["provenance_item_ids"]))
    # Event exports retain original directedness as an attribute; undirected
    # observations are not silently duplicated in the authoritative event layer.
    for event in view["events"]:
        attrs = {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                 for key, value in event.items() if value is not None}
        graph.add_edge(event["source"], event["target"], key=event["edge_id"], **attrs)
    nx.write_graphml(graph, output / "events.graphml")
    nx.write_gexf(graph, output / "events.gexf")
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir())}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    for path in output.iterdir():
        os.chmod(path, 0o600)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--country", action="append", default=[])
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--actor", action="append", default=[])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--min-confidence", type=float, default=0)
    args = parser.parse_args(argv)
    # Nested stage data can be substantially larger than csv's small default.
    csv.field_size_limit(sys.maxsize)
    def records():
        for path in args.inputs:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                yield from csv.DictReader(stream)
    view = aggregate_social_events(records(), relation=args.relation, countries=args.country,
                                   platforms=args.platform, actors=args.actor, start=args.start,
                                   end=args.end, min_confidence=args.min_confidence)
    previous = os.umask(0o077)
    try:
        export_view(view, args.output)
    finally:
        os.umask(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
