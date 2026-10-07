#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
RUN_DIR="${LACLAUGPT_PIPELINE_RUN_DIR:?}"
JOBS_TSV="${LACLAUGPT_PIPELINE_JOBS_TSV:?}"
python3 - "${RUN_DIR}" "${JOBS_TSV}" <<'PY'
import csv, json, pathlib, subprocess, sys
from datetime import datetime, timezone
run_dir=pathlib.Path(sys.argv[1])
jobs_path=pathlib.Path(sys.argv[2])
rows=list(csv.DictReader(jobs_path.open(), delimiter="\t"))
ids=",".join(row["job_id"] for row in rows)
states={}
if ids:
    p=subprocess.run(["sacct","-n","-P","-j",ids,"--format=JobIDRaw,State,ExitCode"],capture_output=True,text=True)
    for line in p.stdout.splitlines():
        parts=line.split("|")
        if len(parts)>=3 and "." not in parts[0]:
            states[parts[0]]={"state":parts[1],"exit_code":parts[2]}
for row in rows:
    row.update(states.get(row["job_id"],{"state":"UNKNOWN","exit_code":"UNKNOWN"}))
success=all(r["state"].startswith("COMPLETED") for r in rows)
payload={"generated_at":datetime.now(timezone.utc).isoformat(),"success":success,"jobs":rows}
(run_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
lines=["# EP24 pipeline run summary","",f"Overall: {'SUCCESS' if success else 'FAILED/INCOMPLETE'}","",
       "| Country | Step | Job | State | Exit |","|---|---:|---:|---|---|"]
for r in rows:
    lines.append(f"| {r['country']} | {r['step']} | {r['job_id']} | {r['state']} | {r['exit_code']} |")
(run_dir/"summary.md").write_text("\n".join(lines)+"\n")
raise SystemExit(0 if success else 1)
PY
