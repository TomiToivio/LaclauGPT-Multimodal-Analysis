"""Synthetic tests for the stage-coverage audit (#306). No database required."""

from __future__ import annotations

from dataclasses import dataclass

import ep24_coverage_audit as audit


@dataclass
class FakeStage:
    number: int
    name: str
    appends: tuple[str, ...]


STAGE = FakeStage(1, "preprocess", ("asr_transcript", "asr_language", "asr_runtime_ms",
                                    "preprocess_status"))


class TestEmptyValue:
    def test_none_and_blank_are_empty(self):
        assert audit.empty_value(None)
        assert audit.empty_value("")
        assert audit.empty_value("   ")
        assert audit.empty_value([])
        assert audit.empty_value({})

    def test_zero_and_false_are_results(self):
        # A stage that legitimately produced 0 must not read as "not started".
        assert not audit.empty_value(0)
        assert not audit.empty_value(False)


class TestAuditStage:
    def test_the_four_states_are_kept_apart(self):
        rows = [
            {"asr_transcript": "text", "preprocess_status": "ok"},      # completed
            {"asr_transcript": "", "preprocess_status": "error"},        # failed
            {},                                                          # not started
            {"asr_transcript": "", "preprocess_status": "ok"},           # claimed but empty
        ]
        result = audit.audit_stage(rows, number=1, name="preprocess",
                                   appends=STAGE.appends, status_columns=("preprocess_status",))
        assert result["completed"] == 1
        assert result["failed"] == 1
        assert result["not_started"] == 1
        assert result["claimed_but_empty"] == 1
        assert result["coverage_pct"] == 25.0

    def test_a_status_column_alone_does_not_count_as_output(self):
        # Writing "ok" is not producing a result.
        rows = [{"preprocess_status": "ok"}]
        result = audit.audit_stage(rows, number=1, name="preprocess",
                                   appends=STAGE.appends, status_columns=("preprocess_status",))
        assert result["completed"] == 0
        assert result["claimed_but_empty"] == 1

    def test_a_stage_with_no_output_columns_is_not_auditable(self):
        result = audit.audit_stage([{}, {}], number=9, name="rdf", appends=(), status_columns=())
        assert result["auditable"] is False
        assert result["coverage_pct"] is None, "0% would be a fabricated finding"

    def test_missing_columns_are_reported(self):
        rows = [{"asr_transcript": "x"}]
        result = audit.audit_stage(rows, number=1, name="preprocess",
                                   appends=STAGE.appends, status_columns=())
        assert "asr_language" in result["missing_columns"]
        assert "asr_transcript" not in result["missing_columns"]


class TestAuditRows:
    def test_language_and_media_facts(self):
        rows = [
            {"asr_language": "fi", "allas_filename": "a.mp4", "asr_transcript": "x"},
            {"asr_language": "FI", "allas_filename": "a.mp4", "asr_transcript": "y"},
            {"asr_language": "", "allas_filename": "b.mp4"},
        ]
        result = audit.audit_rows(rows, [STAGE])
        assert result["asr_language"] == {"fi": 2}
        assert result["distinct_media"] == 2
        assert result["rows_sharing_a_media_key"] == 1

    def test_every_contracted_stage_is_reported(self):
        result = audit.audit_rows([{}], [STAGE, FakeStage(2, "frame", ("frame_analysis_1",))])
        assert [s["stage"] for s in result["stages"]] == [1, 2]
