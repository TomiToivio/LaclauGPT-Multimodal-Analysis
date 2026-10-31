# EP24 log directory and filenames (#324)

The canonical **Roihu Step 1–9 submission path** is
`scripts/roihu/submit_step_common.sh` (called by the per-step submit wrappers).
Set `EP24_LOG_DIR` to an **absolute path** to keep all job logs together;
`LACLAUGPT_EP24_LOG_DIR` is retained as an older alias. The default is
`$LACLAUGPT_MULTIMODAL_PRIVATE_ROOT/logs` (private, outside Git).

Example:

```bash
export EP24_LOG_DIR=/approved/private/ep24/logs
bash scripts/roihu/submit_step_1_preprocess.sh --country poland
tail -f "$EP24_LOG_DIR/preprocess-poland.log"
```

The readable canonical filenames are `<step_name>-<country>.log`, for example
`preprocess-poland.log`, `frame-finland.log`,
`discourse_analysis-portugal.log`. Accepted country names are FI/PL/PT
codes and full names, normalized to lowercase full names. A missing or unknown
country is an error, not an `unknown.log` fallback.

**Concurrent safety:** Each SLURM submission writes both standard output and
standard error into `<step_name>-<country>-<job_id>.log` in the **same**
directory, never a shared write target. The canonical filename is a symlink
to the most recently submitted job's log. Previous attempts remain available
by job ID. Use `tail -f` on the archive filename to follow a specific job.
The link indicates *latest submitted*, not necessarily latest completed.

The directory is created at submission with owner-only permissions.
SLURM captures output from the job's start, including errors before the
Python entrypoint. The job log includes diagnostic output from subprocesses.

## Limitations / migration

This change covers **the canonical per-step submit wrappers and their
`step_runner.sbatch` jobs**. Historical direct submissions of standalone
`step_N_roihu_*.sbatch`, other experimental jobs, Python
`logging.FileHandler` paths and pipeline summary jobs may still have
legacy paths. They are **not** covered by this patch, and should not be
claimed standardized without a separate audit and migration. Avoid invoking
those direct SLURM entrypoints for normal FI/PL/PT production chains.
Existing logs are not moved or deleted.
