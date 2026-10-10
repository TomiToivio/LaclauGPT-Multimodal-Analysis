"""Step 5 sees current-source extraction evidence, not cumulative RAG dumps."""
import pandas as pd

from roihu_postprocess import build_postprocess_prompt


def test_step5_prompt_excludes_cross_record_context_and_keeps_primary_evidence(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_STEP5_MAX_PROMPT_CHARS", "12000")
    row = pd.Series({
        "summary_analysis": "European election candidate makes an appeal " * 50,
        "asr_transcript": "Vote in June. " * 70,
        "ocr_1": "European Elections 2024",
        "researcher_note": "researcher uncertain",
        "rag_context_json": "unrelated speaker " * 10000,
        "vllm_video_raw_output": "other huge output " * 10000,
        "dna_statements_json": "unrelated network " * 10000,
    })
    prompt = build_postprocess_prompt(row)
    assert "European election candidate" in prompt
    assert "Vote in June" in prompt
    assert "European Elections 2024" in prompt
    assert "unrelated speaker" not in prompt
    assert "other huge output" not in prompt
    assert "unrelated network" not in prompt
    assert len(prompt) <= 12000


def test_step5_prompt_does_not_mutate_oversize_original():
    summary = "important " * 8000
    row = pd.Series({"summary_analysis": summary})
    prompt = build_postprocess_prompt(row)
    assert "TRUNCATED" in prompt
    assert row["summary_analysis"] == summary
