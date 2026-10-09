"""Regression checks for the private EP24 human-readable reporter."""
from pathlib import Path

import pandas as pd

from ep24_result_reporting import report_stage_rows
from roihu_storage import StorageConfig


def test_reporter_prints_generated_fields_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LACLAUGPT_ANALYSIS_LOG_DIR", str(tmp_path))
    df = pd.DataFrame([{
        "source_text": "PRIVATE INPUT MUST NOT APPEAR",
        "memory_context_json": "PRIVATE CONTEXT MUST NOT APPEAR",
        "frame_analysis_1": "Generated analysis",
        "frame_analysis_status": "ok",
    }])
    report_stage_rows(2, "finland", df, tmp_path / "stage.csv")
    visible = capsys.readouterr().out
    log_file = tmp_path / "step_02_finland.log"
    assert "Generated analysis" in visible
    assert "Generated analysis" in log_file.read_text()
    assert "PRIVATE INPUT" not in visible
    assert "PRIVATE CONTEXT" not in log_file.read_text()


def test_reporter_can_be_disabled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LACLAUGPT_ANALYSIS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LACLAUGPT_PRINT_ANALYSIS", "0")
    report_stage_rows(4, "poland", pd.DataFrame([{"summary_analysis": "result"}]), tmp_path / "file.csv")
    assert not capsys.readouterr().out
    assert not list(tmp_path.glob("*.log"))


def test_mongo_database_defaults_to_uri_path(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_MONGO_URI", "mongodb://example.invalid:27017/exampleDatabase")
    monkeypatch.delenv("LACLAUGPT_MONGO_DATABASE", raising=False)
    assert StorageConfig.from_env().mongo_database == "exampleDatabase"
    monkeypatch.setenv("LACLAUGPT_MONGO_DATABASE", "overrideDatabase")
    assert StorageConfig.from_env().mongo_database == "overrideDatabase"

def test_review_csv_contains_generated_results_only(tmp_path, monkeypatch):
    import csv
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", str(tmp_path))
    monkeypatch.setenv("LACLAUGPT_EP24_REVIEW_DIR", str(tmp_path / "review"))
    monkeypatch.setenv("LACLAUGPT_ANALYSIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    source = pd.DataFrame([{
        "_storage_id": "synthetic-01",
        "source_text": "PRIVATE ORIGINAL SOURCE",
        "asr_transcript": "Generated synthetic transcript",
        "ocr_1": "Generated OCR",
    }])
    report_stage_rows(1, "finland", source, tmp_path / "cumulative.csv")
    report_stage_rows(1, "finland", source, tmp_path / "cumulative.csv")
    review = tmp_path / "review" / "step_01_finland_1234_review.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["record_id"] == "synthetic-01"
    assert rows[0]["asr_transcript"] == "Generated synthetic transcript"
    assert "source_text" not in rows[0]
    assert "PRIVATE ORIGINAL SOURCE" not in review.read_text()
