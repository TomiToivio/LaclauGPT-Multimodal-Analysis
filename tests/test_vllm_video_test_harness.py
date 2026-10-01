"""Synthetic harness test for experiments/vllm_video_test.py.

No private EP24 material and no GPU: the fixture below is invented, and the test
runs the script with --model-backend stub so no model is loaded. It pins the
properties that matter for the smoke test to be trustworthy on Roihu:

* selection is reproducible for a fixed seed and picks only usable rows;
* the Allas object path is derived from the documented convention;
* source columns are preserved and the experimental columns are appended;
* one bad video does not abort the remaining sample;
* the source CSV on disk is never modified;
* the debug log records the environment and the raw response.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "vllm_video_test.py"


def _load_module():
    """Import the standalone script without requiring a package layout."""
    spec = importlib.util.spec_from_file_location("vllm_video_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_video_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def harness(tmp_path):
    """A synthetic EP24-shaped CSV plus a local Allas mirror with real files."""
    module = _load_module()

    rows = []
    for i in range(1, 11):
        country = "Finland" if i % 2 else "Poland"
        video_id = f"HEPP24-{country[:2].upper()}-{i:03d}"
        allas_filename = f"HEPP24/{country}/researcher{i:02d}/{video_id}.mp4"
        rows.append(
            {
                "country": country,
                "author_username": f"researcher{i:02d}",
                "account_type": "Synthetic",
                "source_type": "TikTok",
                "source_recording": f"{country}-feed-{i:02d}.mp4",
                "video_id": video_id,
                "sequence_number": str(i),
                "political_preference": "",
                "allas_filename": allas_filename,
                "new_entity": "",
                "new_theme": "",
                "video_duration": "12.0",
                "researcher_new_persons": "",
                "researcher_new_themes": "",
                "researcher_note": f"synthetic researcher note {i}",
                "future_added_field": f"upstream-{i}",
            }
        )
    rows.append(
        {
            "country": "Finland",
            "author_username": "researcherbad",
            "account_type": "Synthetic",
            "source_type": "TikTok",
            "source_recording": "bad-feed.mp4",
            "video_id": "",
            "sequence_number": "999",
            "political_preference": "",
            "allas_filename": "",
            "new_entity": "",
            "new_theme": "",
            "video_duration": "",
            "researcher_new_persons": "",
            "researcher_new_themes": "",
            "researcher_note": "unusable row",
            "future_added_field": "still-preserved",
        }
    )
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    mirror = tmp_path / "allas"
    for row in rows:
        if not row["allas_filename"]:
            continue
        path = mirror / row["allas_filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-video-bytes")

    return module, input_csv, mirror, tmp_path


def _run(module, argv):
    return module.main(argv)


def test_selection_is_reproducible_for_a_seed(harness):
    module, input_csv, _, _ = harness

    first = module.select_sample(
        module.load_input_csv(input_csv, module.logging.getLogger("t")),
        5,
        42,
        module.logging.getLogger("t"),
    )
    second = module.select_sample(
        module.load_input_csv(input_csv, module.logging.getLogger("t")),
        5,
        42,
        module.logging.getLogger("t"),
    )
    third = module.select_sample(
        module.load_input_csv(input_csv, module.logging.getLogger("t")),
        5,
        7,
        module.logging.getLogger("t"),
    )
    assert first == second, "same seed must select the same rows"
    assert len(first) == 5
    assert first != third, "a different seed should normally differ"


def test_selection_never_picks_a_row_without_media_identifier(harness):
    module, input_csv, _, _ = harness
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    # Every seed we try must avoid the unusable row (the last one).
    unusable_index = df.index[-1]
    for seed in range(25):
        chosen = module.select_sample(df, 5, seed, module.logging.getLogger("t"))
        assert unusable_index not in chosen


def test_remote_path_uses_canonical_allas_filename(harness):
    module, input_csv, _, _ = harness
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    row = df.iloc[0]
    path = module.derive_remote_path(row, module.DEFAULT_ALLAS_PATH_TEMPLATE)
    assert path == row["allas_filename"]


def test_end_to_end_stub_run_preserves_columns_and_writes_log(harness):
    module, input_csv, mirror, tmp_path = harness
    out_csv = tmp_path / "out.csv"
    log_path = tmp_path / "run.log"
    before = hashlib.sha256(input_csv.read_bytes()).hexdigest()

    rc = _run(module, [
        "--input-csv", str(input_csv),
        "--output-csv", str(out_csv),
        "--log-path", str(log_path),
        "--download-dir", str(tmp_path / "dl"),
        "--fetch-backend", "local",
        "--allas-local-root", str(mirror),
        "--model-backend", "stub",
        "--sample-size", "5",
        "--seed", "1",
    ])

    assert rc == 0
    assert out_csv.is_file()
    out = pd.read_csv(out_csv, dtype=str, keep_default_na=False)

    assert len(out) == 5
    # Every actual source column survives untouched, including future fields
    # unknown to the harness.
    source_columns = list(pd.read_csv(input_csv, nrows=0).columns)
    for column in source_columns:
        assert column in out.columns
    # Experimental columns are appended.
    for column in module.OUTPUT_COLUMNS:
        assert column in out.columns
    assert (out["vllm_video_status"] == "ok").all()
    assert (out["vllm_video_model"] == module.DEFAULT_MODEL).all()
    assert out["vllm_video_analysis"].str.contains("STUB OUTPUT").all()

    # The source CSV is untouched.
    assert hashlib.sha256(input_csv.read_bytes()).hexdigest() == before

    # The debug log records the run.
    log = log_path.read_text(encoding="utf-8")
    assert "hostname" in log
    assert "python_version" in log
    assert "raw_response" in log
    assert "succeeded" in log
    assert set(module.OUTPUT_COLUMNS).issubset(out.columns)
    assert set(out["vllm_video_api"]) == {"direct"}
    assert out["vllm_video_inference_seconds"].map(float).ge(0.0).all()
    assert out["vllm_video_prompt_hash"].str.len().eq(12).all()


def test_one_missing_video_does_not_abort_the_sample(harness):
    module, input_csv, mirror, tmp_path = harness

    # Select a fixed sample, then remove one of its files from the mirror.
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    chosen = module.select_sample(df, 5, 3, module.logging.getLogger("t"))
    victim = df.loc[chosen[0]]
    victim_path = mirror / victim["allas_filename"]
    victim_path.unlink()

    out_csv = tmp_path / "out_skip.csv"
    rc = _run(module, [
        "--input-csv", str(input_csv),
        "--output-csv", str(out_csv),
        "--log-path", str(tmp_path / "skip.log"),
        "--download-dir", str(tmp_path / "dl2"),
        "--fetch-backend", "local",
        "--allas-local-root", str(mirror),
        "--model-backend", "stub",
        "--sample-size", "5",
        "--seed", "3",
    ])

    out = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    assert len(out) == 5, "all five rows must still be written"
    statuses = list(out["vllm_video_status"])
    assert statuses.count("error") == 1
    assert statuses.count("ok") == 4
    failed = out[out["vllm_video_status"] == "error"].iloc[0]
    assert "FileNotFoundError" in failed["vllm_video_error"]
    # Four of the five succeeded, so the run is not a total failure.
    assert rc == 0


def test_script_does_not_touch_pipeline_or_ollama(harness):
    """The experiment must stay isolated from the production pipeline.

    Inspect the parsed module instead of its prose: the docstring names the
    pipeline precisely to record that it is left alone, so a text search would
    test the documentation rather than the behaviour.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    imported: set[str] = set()
    dynamic: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        # A dynamic import by literal name would sidestep the check above.
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name in {"import_module", "__import__"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    dynamic.add(arg.value.split(".")[0])

    assert "ollama" not in imported | dynamic
    for stage in ("roihu_frame", "roihu_summary", "roihu_postprocess", "roihu_populism"):
        assert stage not in imported | dynamic


def test_output_columns_are_unique():
    """The shared Roihu/Laskin CSV contract must not contain duplicate headers."""
    module = _load_module()
    assert len(module.OUTPUT_COLUMNS) == len(set(module.OUTPUT_COLUMNS))


def test_secret_redaction_handles_strings_and_diagnostic_objects():
    module = _load_module()
    assert "hunter2" not in module.redact_sensitive("password=hunter2")
    assert "alice:secret" not in module.redact_sensitive("https://alice:secret@example.test/x")
    assert "finish_reason" in module.redact_sensitive({"finish_reason": "stop"})


def test_structured_output_extracts_human_readable_markdown():
    module = _load_module()
    raw = '{"analysis_markdown":"## Result\\nUseful text","SCROLL":false,"SCROLL_SECONDS":[]}'
    analysis, structured_json, status, error = module.parse_structured_output(raw)
    assert analysis == "## Result\nUseful text"
    assert structured_json == raw
    assert status == "ok"
    assert error == ""


def test_structured_output_requires_analysis_markdown():
    module = _load_module()
    raw = '{"SCROLL":false,"SCROLL_SECONDS":[]}'
    analysis, structured_json, status, error = module.parse_structured_output(raw)
    assert analysis == raw
    assert structured_json == raw
    assert status == "parse_failed"
    assert "analysis_markdown" in error


def test_video_api_aliases_resolve_to_internal_shapes():
    module = _load_module()
    logger = module.logging.getLogger("test-video-api-aliases")
    assert module.resolve_video_api("legacy", logger) == "direct"
    assert module.resolve_video_api("modern", logger) == "mm_processor_kwargs"


def test_legacy_video_request_unpacks_two_value_qwen_result(monkeypatch):
    module = _load_module()

    class Processor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "prompt"

    class FakeQwen:
        @staticmethod
        def process_vision_info(messages, image_patch_size=16):
            return ["image"], ["video"]

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", FakeQwen)
    request = module.prepare_vllm_request(
        [{"role": "user", "content": []}],
        Processor(),
        module.logging.getLogger("test-legacy-video"),
        video_api="legacy",
    )
    assert request["prompt"] == "prompt"
    assert request["multi_modal_data"]["image"] == ["image"]
    assert request["multi_modal_data"]["video"] == ["video"]
    assert "mm_processor_kwargs" not in request


def test_modern_video_request_keeps_qwen_video_metadata_in_mm_data(monkeypatch):
    module = _load_module()

    class Processor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "prompt"

    video_tensor = object()
    metadata = {"fps": 2.0, "total_num_frames": 20}

    class FakeQwen:
        @staticmethod
        def process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        ):
            return None, [(video_tensor, metadata)], {"fps": 2.0}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", FakeQwen)
    request = module.prepare_vllm_request(
        [{"role": "user", "content": []}],
        Processor(),
        module.logging.getLogger("test-modern-video"),
        video_api="modern",
    )

    assert request["multi_modal_data"]["video"] == [(video_tensor, metadata)]
    assert request["mm_processor_kwargs"] == {"fps": 2.0}
    assert "video_metadata" not in request["mm_processor_kwargs"]


def test_prompt_receives_real_researcher_feed_metadata(harness):
    module, input_csv, _, tmp_path = harness
    row = module.load_input_csv(input_csv, module.logging.getLogger("t")).iloc[0]
    args = type("Args", (), {
        "video_min_pixels": 4096,
        "video_max_pixels": 262144,
        "video_total_pixels": 20971520,
    })()
    messages = module.build_video_messages(tmp_path / "clip.mp4", args, row=row)
    text_part = messages[1]["content"][1]["text"]
    assert "EP24 SOURCE METADATA" in text_part
    assert f"- video_id: {row['video_id']}" in text_part
    assert f"- author_username: {row['author_username']}" in text_part
    assert f"- source_recording: {row['source_recording']}" in text_part
    assert f"- researcher_note: {row['researcher_note']}" in text_part
    assert "future_added_field" not in text_part
    assert "authorUniqueId" not in text_part
    assert "scrapedCountry" not in text_part


def test_guided_schema_matches_structured_output_contract():
    module = _load_module()
    assert module.guided_decoding_schema() == module.STRUCTURED_OUTPUT_SCHEMA
    assert "analysis_markdown" in module.guided_decoding_schema()["required"]
