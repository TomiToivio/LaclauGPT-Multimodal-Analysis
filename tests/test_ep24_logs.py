"""#324: one log directory, one filename convention, safe under concurrency."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import ep24_logs


@pytest.fixture()
def log_root(tmp_path, monkeypatch):
    monkeypatch.setenv(ep24_logs.ENV_LOG_DIR, str(tmp_path / "logs"))
    return tmp_path / "logs"


def test_filenames_follow_step_country_with_a_hyphen(log_root):
    assert ep24_logs.log_path("preprocess", "poland").name == "preprocess-poland.log"
    assert ep24_logs.log_path("frame_analysis", "finland").name == "frame_analysis-finland.log"
    assert ep24_logs.log_path("laclau_discourse", "portugal").name == "laclau_discourse-portugal.log"


def test_names_are_normalised_not_trusted(log_root):
    assert ep24_logs.log_path("Preprocess", "POLAND").name == "preprocess-poland.log"
    assert ep24_logs.log_path(" frame-analysis ", " Finland ").name == "frame_analysis-finland.log"


def test_an_unusable_name_raises_instead_of_producing_a_mangled_path(log_root):
    with pytest.raises(ValueError):
        ep24_logs.normalise_step("   ")
    with pytest.raises(ValueError):
        ep24_logs.normalise_step(None)
    with pytest.raises(ValueError):
        ep24_logs.normalise_country("")
    with pytest.raises(ValueError):
        ep24_logs.normalise_step("step/../escape")


def test_the_root_is_configurable_and_created(log_root):
    path = ep24_logs.log_path("preprocess", "poland")
    assert path.parent == log_root
    assert log_root.is_dir()


def test_default_root_is_the_repository_logs_directory(monkeypatch):
    monkeypatch.delenv(ep24_logs.ENV_LOG_DIR, raising=False)
    assert ep24_logs.log_dir().name == "logs"
    assert ep24_logs.log_dir().parent == Path(ep24_logs.__file__).resolve().parent


def test_scheduler_paths_live_in_the_same_directory(log_root):
    paths = ep24_logs.slurm_paths("preprocess", "poland")
    assert Path(paths["output"]).parent == log_root
    assert Path(paths["error"]).parent == log_root
    assert Path(paths["output"]).name == "preprocess-poland.log"
    assert Path(paths["error"]).name == "preprocess-poland.err.log"


def test_opening_again_without_a_new_run_keeps_history(log_root):
    handle = ep24_logs.open_log("preprocess", "finland")
    handle.write("first\n")
    handle.close()
    handle = ep24_logs.open_log("preprocess", "finland")
    handle.write("second\n")
    handle.close()
    text = ep24_logs.log_path("preprocess", "finland").read_text(encoding="utf-8")
    assert "first" in text and "second" in text, "opening must not rotate"


def test_a_second_run_rotates_rather_than_truncating(log_root):
    first = ep24_logs.start_run("preprocess", "poland")
    first.write_text("run one\n", encoding="utf-8")
    second = ep24_logs.start_run("preprocess", "poland")
    assert second == first
    archives = sorted(log_root.glob("preprocess-poland.*.log"))
    assert len(archives) == 1, "the previous run must survive as a timestamped sibling"
    assert archives[0].read_text(encoding="utf-8") == "run one\n"
    assert first.read_text(encoding="utf-8") == ""


def test_open_log_writes_a_header_and_then_appends(log_root):
    ep24_logs.start_run("frame_analysis", "finland")
    handle = ep24_logs.open_log("frame_analysis", "finland", run_header="run 1")
    handle.write("line a\n")
    handle.close()
    handle = ep24_logs.open_log("frame_analysis", "finland", run_header="run 2")
    handle.write("line b\n")
    handle.close()
    text = ep24_logs.log_path("frame_analysis", "finland").read_text(encoding="utf-8")
    assert "line a" in text and "line b" in text


def test_concurrent_writers_do_not_destroy_each_other(log_root):
    """Two processes appending to the same step/country must both leave their lines behind."""
    path = ep24_logs.start_run("preprocess", "portugal")
    path.write_text("", encoding="utf-8")
    code = (
        "import ep24_logs,sys;"
        "h=ep24_logs.open_log('preprocess','portugal');"
        "h.write(sys.argv[1]*200);h.close()"
    )
    env = dict(os.environ)
    procs = [
        subprocess.Popen([sys.executable, "-c", code, marker], env=env,
                         cwd=str(Path(__file__).resolve().parents[1]))
        for marker in ("A", "B")
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    text = path.read_text(encoding="utf-8")
    assert "A" * 200 in text and "B" * 200 in text, "one writer truncated the other"


def test_configure_logging_is_idempotent(log_root):
    import logging

    logger = logging.getLogger("ep24_test_idempotent")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    ep24_logs.configure_logging("preprocess", "poland", logger_name="ep24_test_idempotent")
    ep24_logs.configure_logging("preprocess", "poland", logger_name="ep24_test_idempotent")
    assert len(logger.handlers) == 1, "the same file must not be added twice"
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
