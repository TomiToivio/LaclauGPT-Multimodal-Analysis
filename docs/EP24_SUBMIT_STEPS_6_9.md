# EP24: submit only Steps 6–9 on Roihu

Use `scripts/roihu/submit_steps_6_9.sh` to queue the discourse
analysis (6), discourse network (7), social network (8), and RDF (9)
for Finland, Poland and Portugal **in one invocation**. No Steps 1–5
are queued. The previously processed upstream records must already be
available in the configured dataset and storage.

```bash
# Inspect all 12 proposed submissions without calling sbatch
bash scripts/roihu/submit_steps_6_9.sh --dry-run

# Queue 3 independent country chains (12 jobs total)
bash scripts/roihu/submit_steps_6_9.sh

# Queue only Poland (4 jobs)
bash scripts/roihu/submit_steps_6_9.sh --country pl
```

For each country the chain is **6 → 7 → 8 → 9** with SLURM
`afterok:<previous job id>` dependencies. Countries are independent,
so FI, PL and PT can run simultaneously subject to cluster scheduling.
If an earlier step fails, later steps in that country are held by the
dependency and do not run. Steps do *not* execute simultaneously
within a country because DNA/SNA/RDF may depend on earlier results.

The script uses the existing `submit_step_N_*.sh` wrappers, including
their current environment, resource profiles, logging convention
(`EP24_LOG_DIR`) and full-corpus `--limit 0` behavior. It prints
a tab-separated country/step/job_id/dependency table. A failed
submission stops further scheduling; jobs already submitted remain
scheduled and should be reviewed with `squeue`/`scancel`.
No producer data is cleared or altered by the submission script itself.

**Before submitting:** confirm on Roihu that all four step virtual
environments exist, upstream Steps 1–5 for each country are ready, the
private environment points to the correct dataset, and there is no
previous run that would conflict. This launcher never bypasses
SLURM quotas or revalidates data completion.
