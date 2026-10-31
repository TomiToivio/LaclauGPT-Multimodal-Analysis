"""Contract checks for the restricted EP24 Steps 6-9 launcher."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "roihu" / "submit_steps_6_9.sh"

def run(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True)

def test_bash_parses():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()

def test_all_dry_run_has_only_twelve_jobs_and_independent_country_chains():
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    rows = [line.split("\t") for line in result.stdout.strip().splitlines()]
    assert rows[0] == ["country", "step", "job_id", "dependency"]
    assert len(rows) == 13
    for country in ("finland", "poland", "portugal"):
        block = [row for row in rows[1:] if row[0] == country]
        assert [int(row[1]) for row in block] == [6, 7, 8, 9]
        assert block[0][3] == "-"
        for prev, nxt in zip(block, block[1:]):
            assert nxt[3] == "afterok:" + prev[2]

def test_single_country_and_invalid_value():
    result = run("--country", "pl", "--dry-run")
    assert result.returncode == 0
    rows = result.stdout.strip().splitlines()[1:]
    assert len(rows) == 4
    assert all(row.startswith("poland\t") for row in rows)
    assert run("--country", "sweden", "--dry-run").returncode == 2
