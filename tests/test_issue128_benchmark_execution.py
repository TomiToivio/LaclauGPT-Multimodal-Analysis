"""Regression guards for the #128 benchmark harness (PR #139).

These exist because #139 merged with a harness that had never been executed: the
only guard was a source-text grep, so a script that could not even be imported
passed CI. Everything here executes code rather than asserting that strings
appear in a file.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/roihu/benchmark_issue128_backends.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("benchmark_issue128_backends", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_is_importable_from_the_directory_it_lives_in():
    """The sbatch runs `python3 <path>` from elsewhere.

    Python puts the *script's* directory on sys.path, not the cwd, so a script in
    scripts/roihu/ that imports repo-root modules fails on import unless it adds
    the root itself. This is what actually happened: the harness raised
    ModuleNotFoundError: No module named 'asr_backend' before doing any work.

    Importing the module (without running main) must therefore succeed, and a real
    run must get past every import to the point of complaining about configuration.
    """
    imported = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(HARNESS)!r}, run_name='_probe')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = imported.stdout + imported.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert "No module named 'asr_backend'" not in combined, combined
    assert imported.returncode == 0, combined

    # A real invocation with no manifest must fail on the *missing setting*, i.e.
    # only after every import succeeded -- and it must name the setting, not an
    # import. Run from a different cwd to prove the root insertion is what saves it.
    ran = subprocess.run(
        [sys.executable, str(HARNESS)],
        cwd=Path("/"),
        capture_output=True,
        text=True,
        timeout=300,
        env={"PATH": "/usr/bin:/bin"},
    )
    combined = ran.stdout + ran.stderr
    assert "No module named 'asr_backend'" not in combined, combined
    assert ran.returncode != 0
    assert "LACLAUGPT_BENCH_MANIFEST" in combined, combined


def test_harness_puts_repo_root_on_sys_path():
    harness = _load_harness()
    assert str(ROOT) in sys.path


def test_wer_and_cer_are_none_without_a_reference():
    """An empty reference must not become a number.

    The original divided by max(1, len(ref)), so a missing reference produced the
    hypothesis length and entered the aggregate as a terribly wrong transcript --
    and a group with no references at all crashed on statistics.mean([]).
    """
    harness = _load_harness()
    assert harness.wer("", "some hypothesis text here") is None
    assert harness.cer("", "abc") is None
    assert harness.wer("a b c", "a b c") == 0.0
    assert harness.wer("a b c", "a b d") == pytest.approx(1 / 3)
    assert harness.cer("abc", "abd") == pytest.approx(1 / 3)


def test_summary_reports_unscored_groups_instead_of_crashing():
    harness = _load_harness()
    lines = harness.render_summary(
        2,
        [
            {
                "kind": "asr", "backend": "m", "country": "Finland",
                "wer": "0.500000", "runtime_seconds": "1.0", "gpu_peak_mb": "10.0",
            },
            {
                "kind": "asr", "backend": "m", "country": "Portugal",
                "wer": "", "runtime_seconds": "2.0", "gpu_peak_mb": "20.0",
            },
        ],
    )
    text = "\n".join(lines)
    assert "| asr | m | Finland | 1 | 1 | 0.5000 |" in text
    # Portugal has no reference: it stays visible but contributes no error value.
    assert "| asr | m | Portugal | 1 | 0 | n/a |" in text


def test_summary_survives_an_empty_result_set():
    harness = _load_harness()
    lines = harness.render_summary(0, [])
    assert any("Samples: 0" in line for line in lines)


def test_fieldnames_are_stable_even_with_no_results():
    """`list(results[0])` raised IndexError on an empty run."""
    source = HARNESS.read_text(encoding="utf-8")
    assert "list(results[0])" not in source


def test_easyocr_language_group_rejects_mixed_scripts():
    """The EP24 language list spans latin and cyrillic, which EasyOCR cannot load.

    `easyocr.Reader(EASYOCR_LANGS)` raised
      ValueError: Cyrillic is only compatible with English, ...
    so `LACLAUGPT_OCR_ENGINE=easyocr` failed on load in the benchmark and in
    production Step 1 alike.
    """
    import ocr_backend

    assert ocr_backend.easyocr_language_group(["en", "pl", "pt"]) == "latin"
    assert ocr_backend.easyocr_language_group(["bg"]) == "cyrillic"
    with pytest.raises(ValueError, match="combine scripts"):
        ocr_backend.easyocr_language_group(ocr_backend.EASYOCR_LANGS)
    with pytest.raises(ValueError, match="unsupported"):
        ocr_backend.easyocr_language_group(["en", "xx"])


def test_every_ep24_language_belongs_to_a_declared_script_group():
    import ocr_backend

    declared = {
        lang for members in ocr_backend.EASYOCR_SCRIPT_GROUPS.values() for lang in members
    }
    assert set(ocr_backend.EASYOCR_LANGS) <= declared


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "country",
        "video_path",
        "reference_transcript",
        "reference_transcript_provenance",
        "reference_ocr",
        "reference_ocr_provenance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_rejects_model_generated_reference_text(tmp_path):
    harness = _load_harness()
    manifest = tmp_path / "manifest.csv"
    common = {
        "video_path": "/private/not-read-by-manifest-validation.mp4",
        "reference_ocr": "",
        "reference_ocr_provenance": "",
    }
    _write_manifest(
        manifest,
        [
            {
                **common,
                "sample_id": "fi-1",
                "country": "Finland",
                "reference_transcript": "old whisper transcript",
                "reference_transcript_provenance": "openai-whisper",
            },
            {
                **common,
                "sample_id": "pl-1",
                "country": "Poland",
                "reference_transcript": "",
                "reference_transcript_provenance": "",
            },
            {
                **common,
                "sample_id": "pt-1",
                "country": "Portugal",
                "reference_transcript": "",
                "reference_transcript_provenance": "",
            },
        ],
    )
    with pytest.raises(ValueError, match="human-verified"):
        harness.read_manifest(manifest)


def test_manifest_accepts_human_verified_references_and_empty_unscored_rows(tmp_path):
    harness = _load_harness()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "fi-1",
                "country": "Finland",
                "video_path": "/private/fi.mp4",
                "reference_transcript": "ihmisen tarkistama puhe",
                "reference_transcript_provenance": "human_verified:tomi",
                "reference_ocr": "teksti",
                "reference_ocr_provenance": "researcher_verified",
            },
            {
                "sample_id": "pl-1",
                "country": "Poland",
                "video_path": "/private/pl.mp4",
                "reference_transcript": "",
                "reference_transcript_provenance": "",
                "reference_ocr": "",
                "reference_ocr_provenance": "",
            },
            {
                "sample_id": "pt-1",
                "country": "Portugal",
                "video_path": "/private/pt.mp4",
                "reference_transcript": "fala verificada",
                "reference_transcript_provenance": "manual",
                "reference_ocr": "",
                "reference_ocr_provenance": "",
            },
        ],
    )
    rows = harness.read_manifest(manifest)
    assert [row["country"] for row in rows] == ["Finland", "Poland", "Portugal"]
