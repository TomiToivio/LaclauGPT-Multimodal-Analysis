from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from experiments import vllm_video_test as video


def test_step3_context_contains_every_upstream_field_and_provenance_sections():
    row = pd.Series(
        {
            "video_id": "vid-1",
            "country": "Finland",
            "author_username": "author",
            "researcher_note": "human note",
            "asr_transcript": "spoken words",
            "ocr_1": "visible words",
            "frame_analysis_1": "deep still analysis",
            "custom_prior_field": "must survive",
        }
    )
    context = video.ep24_metadata_context(row)
    for value in (
        "Finland",
        "human note",
        "spoken words",
        "visible words",
        "deep still analysis",
        "must survive",
    ):
        assert value in context
    assert "PRIMARY EVIDENCE" in context
    assert "SOURCE METADATA" in context
    assert "RESEARCHER ANNOTATION" in context
    assert "UPSTREAM MODEL / ENRICHMENT CONTEXT" in context


def test_local_checkpoint_preserves_cumulative_fields_and_is_idempotent(tmp_path: Path):
    source = pd.DataFrame(
        [
            {
                "video_id": "vid-1",
                "country": "Finland",
                "author_username": "author",
                "source_type": "instagram",
                "asr_transcript": "hello",
                "frame_analysis_1": "frame evidence",
                "custom_prior_field": "preserve-me",
            }
        ]
    )
    record = {column: "" for column in video.OUTPUT_COLUMNS}
    record["vllm_video_status"] = "ok"
    record["vllm_video_analysis"] = "temporal analysis"
    record["vllm_video_model"] = video.DEFAULT_MODEL
    record["vllm_video_prompt_sha256"] = video.PROMPT_SHA256

    csv_path = tmp_path / "step3.csv"
    sqlite_path = tmp_path / "step3.sqlite3"
    for _ in range(2):
        out = video.write_local_checkpoint(
            source, [0], [record], csv_path, sqlite_path, video.logging.getLogger("issue132")
        )

    assert out.loc[0, "custom_prior_field"] == "preserve-me"
    assert out.loc[0, "source_type"] == "instagram"
    assert out.loc[0, "vllm_video_analysis"] == "temporal analysis"
    saved = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert saved.loc[0, "custom_prior_field"] == "preserve-me"

    with sqlite3.connect(sqlite_path) as db:
        rows = db.execute("SELECT source_id, row_json FROM step3_video_rows").fetchall()
    assert len(rows) == 1
    payload = json.loads(rows[0][1])
    assert payload["custom_prior_field"] == "preserve-me"
    assert payload["vllm_video_analysis"] == "temporal analysis"


def test_mongo_persistence_uses_patch_upserts_not_document_replacement(monkeypatch):
    calls = []

    class FakeConfig:
        mongo_enabled = True

    class FakeConfigFactory:
        @classmethod
        def from_env(cls):
            return FakeConfig()

    class FakeCollection:
        def update_one(self, query, update, upsert=False):
            calls.append((query, update, upsert))

    class FakeDB(dict):
        def __getitem__(self, key):
            return FakeCollection()

    class FakeMongo:
        def __init__(self, config):
            self.db = FakeDB()

        def collection_name(self, purpose):
            return "ep24_fi_dataframe"

        def close(self):
            pass

    monkeypatch.setattr(video, "StorageConfig", FakeConfigFactory)
    monkeypatch.setattr(video, "MongoStorage", FakeMongo)

    df = pd.DataFrame(
        [
            {
                "video_id": "vid-1",
                "country": "Finland",
                "custom_prior_field": "keep",
                "vllm_video_analysis": "analysis",
                "vllm_video_prompt_sha256": "abc",
            }
        ]
    )
    status = video.persist_mongo_patch(
        df, source_hint="input.csv", model="model", logger=video.logging.getLogger("mongo")
    )
    assert status == "mongo_ok:1"
    assert len(calls) == 1
    query, update, upsert = calls[0]
    assert query["_storage_id"]
    assert upsert is True
    assert "$set" in update
    assert "$setOnInsert" in update
    assert update["$set"]["custom_prior_field"] == "keep"
    assert update["$set"]["vllm_video_analysis"] == "analysis"


def test_step3_production_wrapper_still_processes_all_rows_by_default():
    source = Path("step_3_roihu_video.py").read_text(encoding="utf-8")
    assert 'LACLAUGPT_MAX_ROWS' in source
    assert 'if limit > 0' in source
    assert 'argv.extend(["--sample-size", str(limit)])' in source


def test_step3_keeps_mandatory_original_one_second_skip():
    assert video.REQUIRED_INITIAL_SKIP_SECONDS == 1.0
    assert video.VIDEO_INITIAL_SKIP_SECONDS == 1.0
