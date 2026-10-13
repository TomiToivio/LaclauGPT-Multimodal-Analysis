from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_issue128_step1_asr_uses_trimmed_analysis_clip():
    source = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    assert "prepare_analysis_clip(local_path" in source
    assert "asr.transcribe(str(analysis_clip)" in source
    assert "asr.transcribe(local_path" not in source
    assert "mandatory 1.0s skip" in source
