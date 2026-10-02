from __future__ import annotations

import json

import pandas as pd

from ep24_dna import (
    actor_projection,
    build_prompt_context,
    canonicalize_concept,
    deduplicate_event_rows,
    enrich_statement,
    stable_statement_id,
    statements_to_event_rows,
)


def _row():
    return {
        "_storage_id": "record-1",
        "video_id": "video-1",
        "published_at": "2024-05-01T12:00:00Z",
        "asr_transcript": "Actor A supports clean energy.",
        "summary_analysis": "Actor A argues for clean energy.",
        "laclau_summary_md": "A demand is articulated around energy policy.",
        "entities": '["Actor A"]',
        "themes": '["energy"]',
        "ep24_entity_resolution_json": json.dumps(
            [
                {
                    "decision": "RESOLVED",
                    "surface_form": "Actor A",
                    "normalized_form": "actor a",
                    "canonical_name": "Actor A Canonical",
                    "entity_id": "actor-a-id",
                }
            ]
        ),
    }


def test_context_marks_memory_and_rag_as_non_evidence():
    context, truncated = build_prompt_context(
        _row(),
        memory_items=[
            {
                "kind": "theme",
                "label": "clean energy",
                "evidence_role": "normalization_context_not_source_evidence",
            }
        ],
        rag_items=[
            {
                "stage": "summary",
                "source_record_id": "other",
                "text": "Other actor supports clean energy.",
                "evidence_role": "prior_analysis_context_not_source_evidence",
            }
        ],
    )
    assert not truncated
    assert "CURRENT SOURCE EVIDENCE" in context
    assert "NORMALIZATION ONLY, NOT SOURCE EVIDENCE" in context
    assert "RETRIEVED RAG CONTEXT (NOT SOURCE EVIDENCE)" in context


def test_statement_id_is_deterministic():
    kwargs = dict(
        source_record_id="record-1",
        actor_id_or_name="actor-a-id",
        concept_id_or_label="clean-energy-id",
        agreement=True,
        proposition="Support clean energy",
    )
    assert stable_statement_id(**kwargs) == stable_statement_id(**kwargs)


def test_concept_exact_memory_normalization_is_stable():
    label, concept_id, provenance = canonicalize_concept(
        "Clean Energy",
        country="finland",
        memory_items=[{"label": "clean energy"}],
    )
    assert label == "clean energy"
    assert concept_id
    assert provenance == "researcher_memory_exact"


def test_enriched_binary_statement_is_dna_exportable_when_time_present():
    row = _row()
    actor_lookup = {
        "actor a": {"entity_id": "actor-a-id", "canonical_name": "Actor A Canonical"}
    }
    item = enrich_statement(
        {
            "actor_name": "Actor A",
            "concept_label": "clean energy",
            "proposition": "supports clean energy",
            "stance": "support",
            "agreement": True,
            "evidence_quote": "supports clean energy",
            "evidence_source_fields": ["asr_transcript"],
            "confidence": 0.95,
        },
        row=row,
        source_record_id="record-1",
        country="finland",
        actor_lookup=actor_lookup,
        memory_items=[{"label": "clean energy"}],
        model="test-model",
    )
    assert item["network_layer"] == "discourse"
    assert item["actor_id"] == "actor-a-id"
    assert item["qualifier"] == "positive"
    assert item["statement_time"] == "2024-05-01T12:00:00Z"
    assert item["exportable_to_dna"] is True


def test_ambiguous_statement_remains_in_rich_json_but_not_eventlist():
    item = enrich_statement(
        {
            "actor_name": "Actor A",
            "concept_label": "clean energy",
            "proposition": "mentions clean energy",
            "stance": "unknown",
            "agreement": None,
            "evidence_quote": "clean energy",
            "confidence": 0.6,
        },
        row=_row(),
        source_record_id="record-1",
        country="finland",
        model="test-model",
    )
    assert item["agreement"] is None
    assert item["exportable_to_dna"] is False
    assert statements_to_event_rows([item]) == []


def test_binary_statement_without_timestamp_is_review_only():
    row = _row()
    row["published_at"] = ""
    item = enrich_statement(
        {
            "actor_name": "Actor A",
            "concept_label": "clean energy",
            "proposition": "supports clean energy",
            "stance": "support",
            "agreement": True,
            "evidence_quote": "supports clean energy",
            "confidence": 0.9,
        },
        row=row,
        source_record_id="record-1",
        country="finland",
        model="test-model",
    )
    assert item["exportable_to_dna"] is False


def test_eventlist_uses_rdna_variable_names():
    item = enrich_statement(
        {
            "actor_name": "Actor A",
            "concept_label": "clean energy",
            "proposition": "supports clean energy",
            "stance": "support",
            "agreement": True,
            "evidence_quote": "supports clean energy",
            "confidence": 0.95,
        },
        row=_row(),
        source_record_id="record-1",
        country="finland",
        model="test-model",
    )
    rows = statements_to_event_rows([item])
    assert len(rows) == 1
    assert rows[0]["organization"] == "Actor A"
    assert rows[0]["concept"] == "clean energy"
    assert rows[0]["agreement"] is True
    assert rows[0]["time"] == "2024-05-01T12:00:00Z"
    assert rows[0]["document"] == "video-1"


def test_duplicate_policies_match_dna_semantics():
    base = {
        "organization": "A",
        "concept": "C",
        "agreement": True,
        "document": "doc-1",
        "time": "2024-05-01T12:00:00Z",
    }
    rows = [base, dict(base), {**base, "document": "doc-2"}]
    assert len(deduplicate_event_rows(rows, policy="include")) == 3
    assert len(deduplicate_event_rows(rows, policy="document")) == 2
    assert len(deduplicate_event_rows(rows, policy="month")) == 1
    assert len(deduplicate_event_rows(rows, policy="acrossrange")) == 1


def test_actor_congruence_conflict_and_normalization():
    rows = [
        {"organization": "A", "concept": "C1", "agreement": True},
        {"organization": "A", "concept": "C2", "agreement": False},
        {"organization": "B", "concept": "C1", "agreement": True},
        {"organization": "B", "concept": "C2", "agreement": True},
    ]
    congruence = actor_projection(rows, mode="congruence")
    conflict = actor_projection(rows, mode="conflict")
    subtract = actor_projection(rows, mode="subtract")
    jaccard = actor_projection(rows, mode="congruence", normalization="jaccard")

    assert congruence.at["A", "B"] == 1
    assert conflict.at["A", "B"] == 1
    assert subtract.at["A", "B"] == 0
    assert jaccard.at["A", "B"] == 1 / 3


def test_pipeline_column_preservation_fixture():
    incoming = pd.DataFrame([_row()])
    before = incoming.copy(deep=True)
    incoming["dna_analysis_markdown"] = "analysis"
    incoming["dna_statements_json"] = "[]"
    for column in before.columns:
        assert incoming.at[0, column] == before.at[0, column]
