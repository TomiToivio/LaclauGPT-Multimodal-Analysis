"""Evidence budgets and provenance isolation for EP24 #238-241."""
import pandas as pd
import pytest
from ep24_evidence_packets import evidence_packet


@pytest.mark.parametrize("stage,max_chars", [(6, 24000), (7, 18000), (8, 9000)])
def test_packet_bounded_and_preserves_original_data(stage, max_chars):
    row = pd.Series({
        "video_id": "id", "asr_transcript": "Finnish parliament " * 2000,
        "dna_statements_json": "actor: politician states position",
        "rag_context_json": "SECRET_RAG_" * 3000,
        "memory_context_json": "SECRET_MEMORY_" * 3000,
        "laclau_structured_json": "summary of downstream analysis",
    })
    original = row.copy()
    packet, clipped = evidence_packet(row, stage=stage, max_chars=max_chars)
    assert len(packet) <= max_chars
    assert clipped is True
    assert "SECRET_RAG_" not in packet
    assert "SECRET_MEMORY_" not in packet
    pd.testing.assert_series_equal(row, original)


def test_empty_or_only_retrieval_gives_no_unsupported_relation_evidence():
    packet, clipped = evidence_packet(
        pd.Series({"rag_context_json": "Actor A controls Actor B"}), stage=8
    )
    assert "Actor A" not in packet
    assert clipped is False


def test_multilingual_and_conflicting_quotes_remain_attributed():
    row = pd.Series({
        "asr_transcript": "Emme tue tätä lakia",
        "asr_translation": "We do not support this law",
        "summary_analysis": "Model summary claims support",
        "codebook_context_json": "law: canonical term",
    })
    packet, _ = evidence_packet(row, stage=6)
    assert "[asr_transcript" in packet
    assert "Emme tue tätä lakia" in packet
    assert "We do not support this law" in packet
    assert "Model summary claims support" in packet
    assert "researcher normalization, not source quote" in packet


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        evidence_packet({}, stage=7, max_chars=20)
