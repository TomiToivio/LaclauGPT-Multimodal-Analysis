"""Contract tests for EP24 logging (#324).

The scheduler-side contract lives in the shell submission script; the canonical per-step/per-country
log moved into ``ep24_logging`` when the two halves were reconciled (#329). These tests used to assert
the shell implementation of the canonical log — ``LOG_STEM``, ``--open-mode=append``, a ``ln -sfn``
alias — which no longer exists, so they failed on ``main`` while the behaviour they were protecting had
simply moved. They now assert the split contract: the script names one scheduler log per country in one
directory, and the canonical log file is pinned where it is actually produced.
"""
from pathlib import Path
import re

from ep24_logging import canonical_log_path, normalize_country, normalize_step, slurm_log_path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = (ROOT / "scripts/roihu/roihu_step_profiles.sh").read_text()
SUBMIT = (ROOT / "scripts/roihu/submit_step_common.sh").read_text()


def test_all_steps_have_canonical_names():
    for step in range(1, 10):
        assert re.search(rf"\b{step}\) echo [a-z_]+ ;;", PROFILES)


def test_scheduler_logs_are_country_named_and_in_one_directory():
    # One log directory for Python and scheduler output (#324), configurable, legacy name honoured.
    assert 'LOG_DIR="${EP24_LOG_DIR:-' in SUBMIT
    assert 'export EP24_LOG_DIR="${LOG_DIR}"' in SUBMIT
    assert 'mkdir -p "${LOG_DIR}"' in SUBMIT
    # Per-job scheduler logs carry both the canonical step name and the country.
    assert '--output="${LOG_DIR}/${STEP_NAME}-${COUNTRY}-%j.out"' in SUBMIT
    assert '--error="${LOG_DIR}/${STEP_NAME}-${COUNTRY}-%j.err"' in SUBMIT
    assert '--job-name="ep24-${STEP_NAME}-${COUNTRY}"' in SUBMIT


def test_country_normalization_and_fail_closed():
    # Short forms map onto the canonical slugs the log filenames use.
    for short, canonical in (("fi", "finland"), ("pl", "poland"), ("pt", "portugal")):
        assert f'{short}|' in SUBMIT and f'COUNTRY="{canonical}"' in SUBMIT
    assert 'COUNTRY="${LACLAUGPT_COUNTRY:-all}"' in SUBMIT
    assert 'export LACLAUGPT_COUNTRY="${COUNTRY}"' in SUBMIT
    assert normalize_country("FI") == "finland"


def test_canonical_log_path_is_one_file_per_step_and_country():
    # The canonical log is written by ep24_logging, not by the shell script: assert it where it lives.
    path = canonical_log_path(3, "finland", root=ROOT / "logs-test")
    assert path.name == "video_analysis-finland.log"
    assert canonical_log_path(7, "pt", root=ROOT / "logs-test").name == "discourse_network-portugal.log"
    assert normalize_step(7) == "discourse_network"
    assert slurm_log_path(3, "finland", "12345", root=ROOT / "logs-test").name == (
        "video_analysis-finland-12345.out")


def test_unknown_step_and_country_never_become_a_filename():
    import pytest

    for bad in (42, "42", "", "   "):
        with pytest.raises(ValueError):
            normalize_step(bad)
    for bad in ("atlantis", "", "   "):
        with pytest.raises(ValueError):
            normalize_country(bad)
