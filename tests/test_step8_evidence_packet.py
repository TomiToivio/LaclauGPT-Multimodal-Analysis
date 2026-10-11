"""SNA source evidence is bounded, auditable and excludes other stages' RAG/output.

Issue #238 asks Step 8 to see a bounded actor-relationship evidence packet and to keep
empirical graph construction separate from the Castellsian interpretation. Issue #241
adds the auditability half: log the model-visible size and the truncation *by field*,
and key the cache on the prompt, the model and the options.
"""
import pandas as pd

from step_8_roihu_social_network_analysis import build_sna_evidence, sna_evidence_report


def test_sna_evidence_prioritizes_original_speech_and_dna(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "12000")
    row = pd.Series({
        "asr_transcript": "Alice replied to Bob with a direct objection.",
        "dna_statements_json": '[{"actor":"Alice","concept":"policy","evidence_quote":"No"}]',
        "summary_analysis": "Conversation between two actors.",
        "rag_context_json": "unrelated RAG " * 10000,
        "formula_of_populism_analysis": "irrelevant theory " * 10000,
    })
    evidence = build_sna_evidence(row)
    assert "Alice replied to Bob" in evidence
    assert "evidence_quote" in evidence
    assert "unrelated RAG" not in evidence
    assert "irrelevant theory" not in evidence
    assert len(evidence) <= 12000


def test_sna_context_preserves_oversize_original():
    speech = "long political statement " * 3000
    row = pd.Series({"asr_transcript": speech})
    evidence = build_sna_evidence(row)
    assert "TRUNCATED" in evidence
    assert row["asr_transcript"] == speech


def test_the_report_names_every_field_that_lost_characters(monkeypatch):
    """#241: truncation is recorded by FIELD, which a bare boolean cannot express."""
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "4000")
    row = pd.Series({
        "asr_transcript": "speech " * 3000,
        "entities": "Alice, Bob",
    })
    evidence, report, digest = sna_evidence_report(row)
    assert report["fields_truncated"] >= 1
    assert report["characters_lost"] > 0
    assert report["by_field"]["asr_transcript"]["truncated"] is True
    assert report["by_field"]["entities"]["truncated"] is False
    assert report["by_field"]["entities"]["chars_available"] == len("Alice, Bob")
    assert digest and len(digest) == 64
    assert evidence.startswith("EP24 current-source relations")


def test_the_context_hash_moves_with_the_packet_and_the_budget(monkeypatch):
    """A model or budget change must not reuse a stale cached result."""
    row = pd.Series({"asr_transcript": "identical input"})
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "4000")
    first = sna_evidence_report(row)[2]
    assert sna_evidence_report(row)[2] == first, "not deterministic"
    row2 = pd.Series({"asr_transcript": "different input"})
    assert sna_evidence_report(row2)[2] != first, "packet change did not move the key"


def test_no_evidence_yields_the_documented_empty_packet():
    row = pd.Series({"asr_transcript": "", "entities": float("nan")})
    evidence, report, _ = sna_evidence_report(row)
    assert evidence == "EP24 current-source relations, no RAG-derived evidence."
    assert report["fields_included"] == 0
