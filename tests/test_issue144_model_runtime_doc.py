"""Guards for docs/MODEL_RUNTIME_MATRIX.md (#144).

#144 is documentation-only, so the risk is drift: the doc records each step's
actual runtime/model default as *measured*, and a later commit could change a
default without the doc noticing (or falsify the doc against the code). These
tests read the code and assert the documented facts still hold. They also assert
that step 9 really has no model, since that is an explicit acceptance criterion.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "MODEL_RUNTIME_MATRIX.md"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_doc_exists_and_covers_every_step():
    doc = DOC.read_text(encoding="utf-8")
    for step in range(1, 10):
        assert f"| {step} |" in doc, f"step {step} missing from the matrix"


def test_ollama_steps_share_the_documented_default_model():
    """Steps 2 and 4-8 share the centralized qwen3.8:27b Ollama default."""
    for relative in (
        "roihu_frame.py",
        "roihu_summary.py",
        "roihu_postprocess.py",
        "roihu_populism.py",
        "step_7_roihu_discourse_network_analysis.py",
        "step_8_roihu_social_network_analysis.py",
    ):
        text = _text(relative)
        assert "ollama_model" in text, relative
    models = _text("ep24_models.py")
    assert 'DEFAULT_OLLAMA_MODEL = "qwen3.8:27b"' in models
    assert "LACLAUGPT_MULTIMODAL_MODEL" in models
    assert "qwen3.8:27b" in DOC.read_text(encoding="utf-8")


def test_step3_uses_vllm_with_the_documented_baseline_model():
    text = _text("experiments/vllm_video_test.py")
    assert 'DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"' in text
    assert "vllm" in text.casefold()
    assert "Qwen/Qwen3-VL-32B-Instruct" in DOC.read_text(encoding="utf-8")


def test_step1_specialist_backends_match_the_document():
    asr = _text("asr_backend.py")
    assert 'DEFAULT_ENGINE = os.getenv("LACLAUGPT_ASR_ENGINE", "canary")' in asr
    assert 'DEFAULT_CANARY_MODEL = "nvidia/canary-1b-v2"' in asr

    ocr = _text("ocr_backend.py")
    assert 'DEFAULT_ENGINE = os.getenv("LACLAUGPT_OCR_ENGINE", "paddleocr")' in ocr
    assert 'DEFAULT_PADDLE_VERSION = os.getenv("LACLAUGPT_OCR_MODEL", "PP-OCRv5")' in ocr

    doc = DOC.read_text(encoding="utf-8")
    assert "nvidia/canary-1b-v2" in doc
    assert "PP-OCRv5" in doc


def test_step9_is_recorded_as_deterministic_no_model():
    """Acceptance criterion: step 9 must be explicitly model-free."""
    # The RDF exporter must not import any inference client.
    rdf = _text("roihu_csv_rdf.py")
    for client in ("ollama", "vllm", "transformers", "torch"):
        assert f"import {client}" not in rdf, f"step 9 unexpectedly imports {client}"
    assert "model=none" in rdf
    assert "model=none" in DOC.read_text(encoding="utf-8")


def test_doc_separates_measured_from_estimated():
    doc = DOC.read_text(encoding="utf-8")
    assert "Measured (repository)" in doc
    assert "Estimated" in doc
    # The one real Roihu datapoint must be named as such.
    assert "Qwen/Qwen3-VL-32B-Instruct" in doc and "vLLM on GH200" in doc
