"""Step 2 must never stuff entire Mongo RAG and memory into a frame prompt."""
import sys
from types import ModuleType

import pandas as pd

sys.modules.setdefault('ollama', ModuleType('ollama'))
from roihu_frame import _row_context  # noqa: E402


def test_frame_context_is_bounded_and_prioritizes_direct_evidence(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_FRAME_CONTEXT_MAX_CHARS", "9000")
    row = pd.Series({
        "video_id": "frame-01", "country": "Finland",
        "entities": '["researcher named politician"]',
        "ocr_1": "Visible poster text",
        "asr_transcript": "Spoken primary evidence " * 40,
        "rag_context_json": "OTHER VIDEOS " * 6000,
        "memory_context_json": "OTHER MEMORY " * 4000,
        "asr_translated": "Translation " * 100,
    })
    context, digest = _row_context(row)
    assert "frame-01" in context
    assert "Visible poster text" in context
    assert "Spoken primary evidence" in context
    assert "OTHER VIDEOS" not in context
    assert "OTHER MEMORY" not in context
    assert len(context) < 10000
    assert len(digest) == 64
    assert _row_context(row) == (context, digest)


def test_frame_context_truncates_oversize_asr_without_mutating_source(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_FRAME_CONTEXT_MAX_CHARS", "3000")
    spoken = "FOO " * 10000
    row = pd.Series({"video_id": "v1", "ocr_1": "SIGN", "asr_transcript": spoken})
    context, _ = _row_context(row)
    assert len(context) < 4000
    assert "SIGN" in context
    assert row["asr_transcript"] == spoken
