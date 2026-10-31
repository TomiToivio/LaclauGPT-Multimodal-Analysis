"""Contract tests for EP24 central SLURM logging (#324)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROFILES = (ROOT / "scripts/roihu/roihu_step_profiles.sh").read_text()
SUBMIT = (ROOT / "scripts/roihu/submit_step_common.sh").read_text()

def test_all_steps_have_canonical_names():
    for step in range(1, 10):
        assert re.search(rf"\b{step}\) echo [a-z_]+ ;;", PROFILES)

def test_job_logs_are_country_named_and_in_single_directory():
    assert 'LOG_STEM="${STEP_NAME}-${COUNTRY}"' in SUBMIT
    assert '--output="${LOG_DIR}/${LOG_STEM}-%j.log"' in SUBMIT
    assert '--error="${LOG_DIR}/${LOG_STEM}-%j.log"' in SUBMIT
    assert 'mv -f "${LOG_DIR}/.${LOG_STEM}.latest.$$" "${LOG_DIR}/${LOG_STEM}.log"' in SUBMIT

def test_country_normalization_and_fail_closed():
    for country in ("finland", "poland", "portugal"):
        assert f'COUNTRY={country}' in SUBMIT
    assert 'requires --country finland|poland|portugal' in SUBMIT

def test_job_archives_and_canonical_links():
    assert '--open-mode=append' in SUBMIT
    assert 'ln -sfn' in SUBMIT
    assert 'Job archive:' in SUBMIT
