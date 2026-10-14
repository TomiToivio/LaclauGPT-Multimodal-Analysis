"""SNA source evidence is bounded and excludes other stages' RAG/output."""
import pandas as pd
from step_8_roihu_social_network_analysis import build_sna_evidence


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


def test_sna_evidence_audit_preserves_main_builder(monkeypatch):
    from step_8_roihu_social_network_analysis import sna_evidence_report
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "4000")
    row = pd.Series({"asr_transcript": "speech " * 3000, "entities": "Alice, Bob"})
    original = row.copy()
    evidence, report, digest = sna_evidence_report(row, model="example:1")
    assert evidence == build_sna_evidence(row)
    assert report["by_field"]["asr_transcript"]["truncated"]
    assert report["by_field"]["entities"]["chars_available"] == len("Alice, Bob")
    assert report["characters_lost"] > 0
    assert len(digest) == 64
    assert row.equals(original)


def test_sna_fingerprint_covers_model_and_evidence():
    from step_8_roihu_social_network_analysis import sna_evidence_report
    row = pd.Series({"asr_transcript": "Alice opposed Bob."})
    a = sna_evidence_report(row, model="one")[2]
    assert a != sna_evidence_report(row, model="two")[2]
    assert a != sna_evidence_report(pd.Series({"asr_transcript": "Bob supported Alice."}), model="one")[2]

def _budget_module():
    from step_8_roihu_social_network_analysis import step8_max_evidence_chars
    return step8_max_evidence_chars


def test_step8_honours_the_shared_context_budget(monkeypatch):
    """Steps 6 and 7 read LACLAUGPT_STEP<stage>_MAX_CONTEXT_CHARS; Step 8 used to shadow it.

    Setting the shared name did nothing for Step 8, so the stage was not in fact configurable
    through the variable the rest of the programme reads.
    """
    monkeypatch.delenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", raising=False)
    row = pd.Series({"asr_transcript": "speech " * 3000})
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", "12000")
    wide = build_sna_evidence(row)
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", "4000")
    narrow = build_sna_evidence(row)
    assert len(narrow) < len(wide)


def test_step8_default_budget_is_unchanged_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", raising=False)
    monkeypatch.delenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", raising=False)
    assert _budget_module()() == 12000


def test_step8_historical_variable_still_works(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", raising=False)
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "5000")
    assert _budget_module()() == 5000


def test_step8_shared_variable_takes_precedence(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", "9000")
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", "5000")
    assert _budget_module()() == 9000


def test_step8_budget_keeps_its_floor_and_survives_a_bad_value(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", raising=False)
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", "10")
    assert _budget_module()() == 3000
    monkeypatch.setenv("LACLAUGPT_STEP8_MAX_CONTEXT_CHARS", "not-a-number")
    assert _budget_module()() == 12000
