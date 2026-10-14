#!/usr/bin/env python3
"""Aggregate Step 8 event edges without mixing relation families.

Input is a cumulative Step 8 CSV; outputs a de-duplicated event list and
typed actor-pair counts, retaining each source/document and evidence quote.
Run after all country-level Stage 8 jobs have finished.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = []
    seen = set()
    with args.input_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sna_status") != "ok":
                continue
            for edge in json.loads(row.get("sna_edges_json") or "[]"):
                family = str(edge.get("relation_type", "")).casefold()
                if family not in {"communication_mention", "mention", "reply",
                                  "quotation", "repost", "direct_interaction",
                                  "institutional_affiliation", "affiliation"}:
                    continue
                identity = (edge.get("canonical_item_id"), edge.get("source"),
                            edge.get("target"), family, edge.get("evidence_quote"))
                if identity in seen:
                    continue
                seen.add(identity)
                events.append({
                    "document_id": edge.get("canonical_item_id", ""),
                    "source_id": edge.get("source", ""),
                    "source_label": edge.get("source_label", ""),
                    "target_id": edge.get("target", ""),
                    "target_label": edge.get("target_label", ""),
                    "relation_type": family,
                    "network_layer": "institutional" if family in {"affiliation", "institutional_affiliation"} else "communication",
                    "timestamp": edge.get("timestamp", ""),
                    "evidence_quote": edge.get("evidence_quote", ""),
                    "edge_id": edge.get("edge_id", ""),
                })
    fields = ["document_id","source_id","source_label","target_id","target_label",
              "relation_type","network_layer","timestamp","evidence_quote","edge_id"]
    with (args.output_dir / "sna_events.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(events)
    counts = Counter((e["source_id"],e["target_id"],e["relation_type"],e["network_layer"]) for e in events)
    with (args.output_dir / "sna_typed_pairs.csv").open("w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id","target_id","relation_type","network_layer","event_count"])
        for key, count in sorted(counts.items()):
            w.writerow([*key,count])
    print(f"Aggregated {len(events)} unique events into {len(counts)} typed actor pairs")

if __name__ == "__main__":
    main()
