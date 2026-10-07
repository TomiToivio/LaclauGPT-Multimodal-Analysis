from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROIHU = ROOT / "scripts" / "roihu"

STEP_NAMES = {
    1: "preprocess",
    2: "frame",
    3: "video",
    4: "summary",
    5: "postprocess",
    6: "discourse_analysis",
    7: "discourse_network_analysis",
    8: "social_network_analysis",
    9: "rdf",
}

def _write_country(root: Path, country: str, n: int = 15) -> None:
    path = root / f"ep24_{country}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "allas_filename", "country"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "video_id": f"{country}-{i:02d}",
                "allas_filename": f"{country}/{i:02d}.mp4",
                "country": country,
            })

def test_every_step_has_setup_submit_and_requirements():
    for step, name in STEP_NAMES.items():
        assert (ROIHU / f"setup_step_{step}_{name}.sh").is_file()
        assert (ROIHU / f"submit_step_{step}_{name}.sh").is_file()
        req_name = name.replace("_", "-")
        assert (ROOT / "requirements" / f"roihu-step{step}-{req_name}.txt").is_file()

def test_step9_profile_is_cpu_only():
    profile = (ROIHU / "roihu_step_profiles.sh").read_text()
    assert '9) echo "small|4|16G||04:00:00"' in profile
    assert "1|2|3|4|5|6|7|8" in profile

def test_test_manifest_is_exact_and_reproducible(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for country in ("finland", "poland", "portugal"):
        _write_country(source, country)

    selected_sets = []
    for run_name in ("a", "b"):
        run = tmp_path / run_name
        result = subprocess.run(
            [
                sys.executable,
                str(ROIHU / "prepare_pipeline_run.py"),
                "--test",
                "--input-root",
                str(source),
                "--run-dir",
                str(run),
                "--seed",
                "209",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        effective = Path(result.stdout.strip())
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["sample_size"] == 30
        assert manifest["countries"] == ["finland", "poland", "portugal"]
        counts = {}
        for item in manifest["sample"]:
            counts[item["country"]] = counts.get(item["country"], 0) + 1
        assert counts == {"finland": 10, "poland": 10, "portugal": 10}
        assert all(sum(1 for _ in csv.DictReader((effective / f"ep24_{c}.csv").open())) == 10
                   for c in manifest["countries"])
        selected_sets.append([(x["country"], x["record_id"]) for x in manifest["sample"]])

    assert selected_sets[0] == selected_sets[1]

def test_full_manifest_discovers_all_country_csvs(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for country in ("finland", "poland", "portugal", "germany"):
        _write_country(source, country, 1)
    run = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            str(ROIHU / "prepare_pipeline_run.py"),
            "--full",
            "--input-root",
            str(source),
            "--run-dir",
            str(run),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["countries"] == ["finland", "germany", "poland", "portugal"]
    assert manifest["sample"] == []

def test_pipeline_modes_are_explicit():
    script = (ROIHU / "run_pipeline.sh").read_text()
    assert "--test|--full" in script
    assert "mutually exclusive" in script
    assert "afterok:" in script
    assert "afterany:" in script
    assert "LACLAUGPT_MONGO_ENABLED=0" in script


def test_summary_tracks_each_video_at_each_step(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    outputs = run / "outputs" / "finland"
    outputs.mkdir(parents=True)
    records = [
        {
            "country": "finland",
            "source_index": 0,
            "record_key": "video_id",
            "record_id": "fi-1",
        },
        {
            "country": "finland",
            "source_index": 1,
            "record_key": "video_id",
            "record_id": "fi-2",
        },
    ]
    (run / "manifest.json").write_text(
        json.dumps({"mode": "test", "records": records, "sample": records})
    )
    jobs = run / "jobs.tsv"
    jobs.write_text("country\\tstep\\tjob_id\\nfinland\\t1\\t101\\n")
    with (outputs / "step_01_preprocess.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id"])
        writer.writeheader()
        writer.writerow({"video_id": "fi-1"})
        writer.writerow({"video_id": "fi-2"})

    fake_sacct = tmp_path / "sacct"
    fake_sacct.write_text("#!/bin/sh\\necho '101|COMPLETED|0:0'\\n")
    fake_sacct.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + __import__("os").environ["PATH"])
    monkeypatch.setenv("LACLAUGPT_EP24_OUTPUT_ROOT", str(run / "outputs"))

    result = subprocess.run(
        [
            sys.executable,
            str(ROIHU / "summarize_pipeline_run.py"),
            str(run),
            str(jobs),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    summary = json.loads((run / "summary.json").read_text())
    step1 = [item for item in summary["video_steps"] if item["step"] == 1]
    assert {item["status"] for item in step1} == {"present"}
    step2 = [item for item in summary["video_steps"] if item["step"] == 2]
    assert {item["status"] for item in step2} == {"no_output"}
