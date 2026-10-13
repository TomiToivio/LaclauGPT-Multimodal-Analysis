"""Tests for the RDF / knowledge-graph export stage (issue #5, step `rdf`).

These run against a tiny **synthetic** EP24-shaped fixture built in ``tmp_path``,
so they need no private data, no CSC allocation and no network. The fixture
deliberately includes a column the exporter does not know about, because the
"retain all legacy fields" requirement is the one most likely to regress.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "roihu_rdf.py"

LG = "https://w3id.org/laclau" + "gpt/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: A synthetic row using the exact column names the five legacy stages produce,
#: plus one unknown column so lossless preservation is testable.
SYNTHETIC_ROW = {
    "authorUniqueId": "synth_author",
    "videoId": "9000000000000000001",
    "language": "fi",
    "scrapedCountry": "FI",
    "frame_files": (
        "./Keyframes/TikTok/synth_author/9000000000000000001/1.jpg,"
        "./Keyframes/TikTok/synth_author/9000000000000000001/2.jpg"
    ),
    "whisperResult": "SYNTHETIC transcript text.",
    "whisper_transcript": "SYNTHETIC transcript text.",
    "whisper_language": "fi",
    "whisper_translated": "SYNTHETIC translated text.",
    "ocr_1": "SYNTHETIC overlay one",
    "ocr_2": "SYNTHETIC overlay two",
    "ocr_3": "",
    "ocr_4": "",
    "ocr_5": "",
    "ocr_6": "",
    "frame_analysis_1": "SYNTHETIC frame one analysis",
    "frame_analysis_2": "SYNTHETIC frame two analysis",
    "frame_analysis_3": "",
    "frame_analysis_4": "",
    "frame_analysis_5": "",
    "frame_analysis_6": "",
    "metadata": "SYNTHETIC metadata block",
    "authorNickname": "Synthetic Author",
    "authorSignature": "synth",
    "videoCreated": "1759296000",
    "videoDescription": "SYNTHETIC description #tag",
    "videoDuration": "30",
    "videoCommentCount": "4",
    "videoDiggCount": "10",
    "videoPlayCount": "100",
    "videoShareCount": "2",
    "summary_analysis": "SYNTHETIC summary analysis.",
    "entities": "SYNTHETIC Entity A, SYNTHETIC Entity B",
    "topics": "labour, regulation",
    "positive": "SYNTHETIC Entity A",
    "neutral": "",
    "negative": "SYNTHETIC Entity B",
    "video_filename": "synth_author/9000000000000000001",
    "formula_of_populism_analysis": "SYNTHETIC populism analysis.",
    "formula_of_populism_us": "the people^hope",
    "formula_of_populism_frontier": "the elite^anger",
    "a_unknown_legacy_column": "must survive",
}


@pytest.fixture()
def runtime(tmp_path: Path) -> Path:
    """A private-runtime-like working directory with a synthetic CSV."""
    (tmp_path / "csv").mkdir()
    (tmp_path / "logs").mkdir()
    target = tmp_path / "csv" / "tiktok_fi.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SYNTHETIC_ROW))
        writer.writeheader()
        writer.writerow(SYNTHETIC_ROW)
        second = dict(SYNTHETIC_ROW)
        second["videoId"] = "9000000000000000002"
        second["a_unknown_legacy_column"] = "second"
        writer.writerow(second)
    return tmp_path


def run_export(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def parse(turtle: Path):
    """Parse the export with rdflib, or skip.

    rdflib is an OPTIONAL validation extra, never a runtime dependency: the
    exporter itself is stdlib-only so a batch job cannot fail on an import. CI
    installs only ruff + pytest, so this must skip rather than raise — a
    module-scope import here fails collection and takes the whole suite down.
    """
    rdflib = pytest.importorskip(
        "rdflib",
        reason="optional validation extra; the exporter itself needs no RDF library",
    )
    graph = rdflib.Graph()
    graph.parse(turtle, format="turtle")
    return graph


# --------------------------------------------------------------------------
# the stage runs and produces parseable RDF
# --------------------------------------------------------------------------


def test_export_runs_and_writes_turtle_and_summary(runtime: Path) -> None:
    result = run_export(runtime, "--language", "fi")
    assert result.returncode == 0, result.stderr
    assert (runtime / "rdf" / "ep24_fi.ttl").exists()
    assert (runtime / "rdf" / "summary_fi.txt").exists()


def test_output_parses_as_real_rdf(runtime: Path) -> None:
    """A file that merely looks like Turtle is not an export."""
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    assert len(graph) > 0


def test_missing_input_is_skipped_not_fatal(runtime: Path) -> None:
    """One absent language must not fail the batch."""
    result = run_export(runtime, "--language", "sv")
    assert result.returncode == 0
    assert "no input file" in result.stdout


def test_dry_run_writes_nothing(runtime: Path) -> None:
    result = run_export(runtime, "--language", "fi", "--dry-run")
    assert result.returncode == 0
    assert not (runtime / "rdf" / "ep24_fi.ttl").exists()


def test_sample_limits_rows(runtime: Path) -> None:
    result = run_export(runtime, "--language", "fi", "--sample", "1")
    assert result.returncode == 0
    assert "1 row(s)" in result.stdout


# --------------------------------------------------------------------------
# classes use the standard RDF vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        "Document",
        "Actor",
        "Frame",
        "FrameAnalysis",
        "ScreenText",
        "Transcript",
        "Summary",
        "Topic",
        "Entity",
        "SentimentTarget",
        "SentimentAssessment",
        "PopulismAnalysis",
        "PopulismElement",
        "LegacyField",
    ],
)
def test_expected_class_is_asserted(runtime: Path, cls: str) -> None:
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")
    assert (None, rdflib.URIRef(RDF_NS + "type"), rdflib.URIRef(LG + cls)) in graph, cls


# --------------------------------------------------------------------------
# legacy compatibility: nothing may be dropped
# --------------------------------------------------------------------------


def test_every_legacy_column_survives_into_the_graph(runtime: Path) -> None:
    """The acceptance criterion "retains all legacy fields", checked mechanically."""
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")

    emitted = {str(o) for o in graph.objects(None, rdflib.URIRef(LG + "columnName"))}
    assert "a_unknown_legacy_column" in emitted, (
        "an unrecognised legacy column must be preserved as a LegacyField, "
        "or the export silently drops research data"
    )


def test_unknown_legacy_values_are_preserved_verbatim(runtime: Path) -> None:
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")
    values = {str(o) for o in graph.objects(None, rdflib.URIRef(LG + "columnValue"))}
    assert {"must survive", "second"} <= values


# --------------------------------------------------------------------------
# provenance and derivation
# --------------------------------------------------------------------------


def test_provenance_is_attached(runtime: Path) -> None:
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")
    assert any(graph.objects(None, rdflib.URIRef(LG + "runId")))
    assert any(graph.objects(None, rdflib.URIRef(LG + "stage")))


def test_derivation_distinguishes_observed_from_model(runtime: Path) -> None:
    """A platform fact and a model claim must be tellable apart."""
    assert run_export(runtime, "--language", "fi").returncode == 0
    graph = parse(runtime / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")
    values = {str(o) for o in graph.objects(None, rdflib.URIRef(LG + "derivation"))}
    assert {"observed", "model_derived"} <= values


# --------------------------------------------------------------------------
# public-repository safety
# --------------------------------------------------------------------------


def test_no_private_paths_in_output(runtime: Path) -> None:
    assert run_export(runtime, "--language", "fi").returncode == 0
    text = (runtime / "rdf" / "ep24_fi.ttl").read_text(encoding="utf-8")
    for pattern in ("/home/", "/mnt/", "/scratch/", "BEGIN PRIVATE KEY"):
        assert pattern not in text, pattern


def test_declared_namespace_matches_the_sibling_implementation() -> None:
    """One shared vocabulary, not a second one invented here."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "https://w3id.org/laclau" + "gpt/" in text


def test_stage_does_not_run_inside_the_pipeline_stages() -> None:
    """The export is appended; it must not be spliced into the five stages.

    The stages were renamed `puhti_*.py` -> `roihu_*.py` on main after this stage
    landed (commits c3d59aa..c37e389). The check follows the current names and
    skips any that are absent, so a future rename cannot turn a design assertion
    into a FileNotFoundError.
    """
    stages = [
        "roihu_preprocess.py",
        "roihu_frame.py",
        "roihu_summary.py",
        "roihu_postprocess.py",
        "roihu_populism.py",
        # historical names, in case a branch still carries them
        "puhti_preprocess.py",
        "puhti_frame.py",
        "puhti_summary.py",
        "puhti_postprocess.py",
        "puhti_populism.py",
    ]
    checked = 0
    for name in stages:
        path = ROOT / name
        if not path.exists():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        assert "roihu_rdf" not in text, f"{name} must not import the export stage"
    assert checked >= 5, f"expected to check the five stages, checked {checked}"


@pytest.fixture()
def runtime_asr_schema(tmp_path: Path) -> Path:
    """Synthetic row shaped like the active #128 Step-1 producer, not legacy Whisper."""
    (tmp_path / "csv").mkdir()
    (tmp_path / "logs").mkdir()
    row = {
        "video_id": "SYNTH-FI-ASR-0001",
        "country": "Finland",
        "author_username": "synth_asr_author",
        "account_type": "Synthetic",
        "source_type": "TikTok",
        "source_recording": "synthetic-recording",
        "sequence_number": "1",
        "political_preference": "",
        "allas_filename": "synthetic.mp4",
        "new_entity": "",
        "new_theme": "",
        "video_duration": "12.0",
        "researcher_new_persons": "",
        "researcher_new_themes": "",
        "researcher_note": "",
        "language": "fi",
        "frame_file": "./Keyframes/tiktok/synth_asr_author/SYNTH-FI-ASR-0001/frame_t1.0s.jpg",
        "frame_timestamp_seconds": "1.0",
        "ocr_1": "SYNTHETIC screen text",
        "ocr_backend": "paddleocr",
        "ocr_model": "PP-OCRv5",
        "asr_transcript": "SYNTHETIC alkuperäinen puhe.",
        "asr_language": "fi",
        "asr_translated": "SYNTHETIC translated speech.",
        "asr_backend": "canary",
        "asr_model": "nvidia/canary-1b-v2",
        "frame_analysis_1": "SYNTHETIC active frame analysis",
        "summary_analysis": "SYNTHETIC active summary",
    }
    target = tmp_path / "csv" / "tiktok_fi.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return tmp_path


def test_active_asr_schema_emits_typed_transcript(runtime_asr_schema: Path) -> None:
    """Regression for #128: the primary asr_* path must not fall into legacy fallback."""
    result = run_export(runtime_asr_schema, "--language", "fi")
    assert result.returncode == 0, result.stderr
    graph = parse(runtime_asr_schema / "rdf" / "ep24_fi.ttl")
    rdflib = pytest.importorskip("rdflib")

    transcript_type = rdflib.URIRef(LG + "Transcript")
    rdf_type = rdflib.URIRef(RDF_NS + "type")
    nodes = list(graph.subjects(rdf_type, transcript_type))
    assert len(nodes) == 1
    node = nodes[0]

    values = lambda predicate: {str(v) for v in graph.objects(node, rdflib.URIRef(LG + predicate))}
    assert values("text") == {"SYNTHETIC translated speech."}
    assert values("originalText") == {"SYNTHETIC alkuperäinen puhe."}
    assert values("translatedText") == {"SYNTHETIC translated speech."}
    assert values("language") == {"fi"}
    assert values("asrBackend") == {"canary"}
    assert values("asrModel") == {"nvidia/canary-1b-v2"}
