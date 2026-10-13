"""End-to-end demo of the Step 7 DNA export layer on a synthetic fixture.

Not a test (it writes files): run it to inspect the real artefacts and confirm
the rDNA compatibility path. No model, Mongo, Redis or R required.

    python3 examples/step7_dna_demo.py [outdir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_dna as dna  # noqa: E402
import step_7_roihu_discourse_network_analysis as step7  # noqa: E402


def synthetic_statements() -> list[dict]:
    """Four actors, three concepts, deliberately mixed stances."""
    rows = [
        ("Party A", "climate policy", 1, "2024-05-01T09:00:00+00:00"),
        ("Party A", "nuclear energy", 1, "2024-05-01T09:00:00+00:00"),
        ("Party B", "climate policy", 0, "2024-05-02T09:00:00+00:00"),
        ("Party B", "nuclear energy", 1, "2024-05-02T09:00:00+00:00"),
        ("Party C", "climate policy", 1, "2024-05-03T09:00:00+00:00"),
        ("Party C", "tax relief", None, "2024-05-03T09:00:00+00:00"),  # ambiguous
        ("Party D", "climate policy", 1, "2024-05-04T09:00:00+00:00"),
        ("Party D", "tax relief", 1, "2024-05-04T09:00:00+00:00"),
    ]
    out = []
    for actor, concept, agreement, when in rows:
        out.append({
            "statement_id": dna.statement_id(
                source_record_id="SYNTH-DEMO", actor=actor, concept=concept,
                agreement=agreement, date_time=when,
            ),
            "source_record_id": "SYNTH-DEMO",
            "document_id": "SYNTH-DEMO",
            "network_layer": "discourse",
            "organization": actor,
            "actor_id": f"actor:{actor.casefold().replace(' ', '-')}",
            "concept": concept,
            "concept_id": f"concept:{concept.replace(' ', '-')}",
            "proposition": f"{actor} takes a position on {concept}.",
            "agreement": agreement,
            "date_time": when,
            "evidence_quote": f"synthetic quote about {concept}",
            "confidence": 0.9,
            "provenance": "model_derived",
        })
    return out


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/step7_demo")
    outdir.mkdir(parents=True, exist_ok=True)
    statements = synthetic_statements()

    frame = pd.DataFrame([{
        "video_id": "SYNTH-DEMO",
        "_storage_id": "SYNTH-DEMO",
        "allas_filename": "https://example.invalid/SYNTH.mp4",
        "country": "Finland",
        "dna_statements_json": json.dumps(statements, ensure_ascii=False),
    }])

    counts = dna.summarize(statements)
    print("=== statement summary ===")
    print(json.dumps(counts, indent=2))

    written = step7.write_exports(frame, "finland", outdir / "ep24_finland.csv")
    print("\n=== artefacts written ===")
    for name, path in sorted(written.items()):
        print(f"  {name:26s} {path}")

    print("\n=== rDNA event list (binary statements only) ===")
    events = dna.read_event_list_csv(written["event_list"])
    for event in events:
        print(f"  {event['organization']:9s} | {event['concept']:15s} | agreement={event['agreement']} | {event['date_time']}")
    print(f"  ({counts['uncertain_count']} ambiguous statement(s) kept in JSON, excluded from the binary export)")

    print("\n=== actor x concept two-mode (subtract) ===")
    network = dna.two_mode_network(statements, qualifier_aggregation="subtract")
    print(f"  rows={network['rows']}")
    print(f"  cols={network['columns']}")
    for name, row in zip(network["rows"], network["values"]):
        print(f"  {name}: {row}")

    print("\n=== actor congruence (raw, then jaccard) ===")
    congruence = dna.actor_congruence_network(statements)
    normalized = dna.apply_normalization(congruence, statements=statements, normalization="jaccard")
    print(f"  raw      {congruence}")
    print(f"  jaccard  {[(a, b, round(w, 4)) for a, b, w in normalized]}")

    print("\n=== actor conflict ===")
    print(f"  {dna.actor_conflict_network(statements)}")

    print("\n=== signed concepts (concept|qualifier) ===")
    print(f"  {dna.signed_concepts(statements)}")

    print("\n=== generated rDNA importer (first 12 lines) ===")
    for line in Path(written["importer"]).read_text(encoding="utf-8").splitlines()[:12]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
