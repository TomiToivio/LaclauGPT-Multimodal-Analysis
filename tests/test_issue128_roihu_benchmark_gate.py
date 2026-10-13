from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_issue128_benchmark_harness_requires_three_smoke_countries_and_private_manifest():
    source = (ROOT / "scripts/roihu/benchmark_issue128_backends.py").read_text(encoding="utf-8")
    assert '{"finland", "poland", "portugal"}' in source
    assert 'LACLAUGPT_BENCH_MANIFEST' in source
    assert 'prepare_analysis_clip' in source
    assert 'frame_at_analysis_start' in source
    assert 'DEFAULT_ASR = ("canary", "parakeet", "qwen3-asr", "faster-whisper")' in source
    assert 'DEFAULT_OCR = ("paddleocr", "easyocr")' in source


def test_issue128_benchmark_sbatch_targets_roihu_gh200():
    text = (ROOT / "scripts/roihu/issue128_backend_benchmark.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "LACLAUGPT_BENCH_MANIFEST" in text
    assert "LaclauGPT-Private" in text
