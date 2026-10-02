from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_LIST_FILES = (
    "roihu_preprocess.py",
    "roihu_frame.py",
    "roihu_summary.py",
    "roihu_postprocess.py",
    "roihu_rdf.py",
    "step_7_roihu_discourse_network_analysis.py",
    "step_8_roihu_social_network_analysis.py",
)


def test_bulgarian_is_in_active_roihu_language_defaults():
    for relative in ACTIVE_LIST_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "bg" in text, f"Bulgarian missing from {relative}"


def test_bulgarian_is_enabled_for_easyocr_fallback():
    text = (ROOT / "ocr_backend.py").read_text(encoding="utf-8")
    assert "easyocr.Reader" in text
    assert '"bg"' in text


def test_country_runtime_knows_bulgaria():
    text = (ROOT / "roihu_codebooks.py").read_text(encoding="utf-8")
    assert '"BG"' in text
    assert '"languages": ["bg", "en"]' in text

    enrich = (ROOT / "roihu_enrich.py").read_text(encoding="utf-8")
    assert '"BG": ("ep24_bg.csv", "bg")' in enrich
