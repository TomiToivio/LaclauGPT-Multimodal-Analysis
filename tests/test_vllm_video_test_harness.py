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
        rows.append(
            {
                "language": "en",
                "authorUniqueId": f"user{i:02d}",
                "scrapedCountry": "Finland" if i % 2 else "Poland",
                "videoId": f"70000000000000{i:02d}",
                "authorNickname": f"nick{i}",
                "videoDescription": f"synthetic row {i}",
            }
        )
    # One row that cannot be fetched: it has no videoId, so it is not "usable"
    # and must never be selected.
    rows.append(
        {
            "language": "en",
            "authorUniqueId": "userbad",
            "scrapedCountry": "Finland",
            "videoId": "",
            "authorNickname": "bad",
            "videoDescription": "unusable row",
        }
    )
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    mirror = tmp_path / "allas"
    for row in rows:
        if not row["videoId"]:
            continue
        object_dir = (
            mirror / "Scraper" / "TikTok" / "Videos" / row["scrapedCountry"]
            / row["authorUniqueId"]
        )
        object_dir.mkdir(parents=True, exist_ok=True)
        (object_dir / f"{row['videoId']}.mp4").write_bytes(b"synthetic-video-bytes")

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


def test_selection_never_picks_a_row_without_a_video_id(harness):
    module, input_csv, _, _ = harness
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    # Every seed we try must avoid the unusable row (the last one).
    unusable_index = df.index[-1]
    for seed in range(25):
        chosen = module.select_sample(df, 5, seed, module.logging.getLogger("t"))
        assert unusable_index not in chosen


def test_remote_path_follows_the_documented_convention(harness):
    module, input_csv, _, _ = harness
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    row = df.iloc[0]
    path = module.derive_remote_path(row, module.DEFAULT_ALLAS_PATH_TEMPLATE)
    assert path == (
        f"Scraper/TikTok/Videos/{row['scrapedCountry']}/"
        f"{row['authorUniqueId']}/{row['videoId']}.mp4"
    )


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
    # Original columns survive untouched.
    for column in ("language", "authorUniqueId", "scrapedCountry", "videoId",
                   "authorNickname", "videoDescription"):
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


def test_one_missing_video_does_not_abort_the_sample(harness):
    module, input_csv, mirror, tmp_path = harness

    # Select a fixed sample, then remove one of its files from the mirror.
    df = module.load_input_csv(input_csv, module.logging.getLogger("t"))
    chosen = module.select_sample(df, 5, 3, module.logging.getLogger("t"))
    victim = df.loc[chosen[0]]
    victim_path = (
        mirror / "Scraper" / "TikTok" / "Videos" / victim["scrapedCountry"]
        / victim["authorUniqueId"] / f"{victim['videoId']}.mp4"
    )
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
