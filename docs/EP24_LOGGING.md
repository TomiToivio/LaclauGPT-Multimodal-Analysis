# EP24 pipeline logs: one directory, one convention (#324)

Every step and country writes to a single log directory under a predictable name:

```
<log_dir>/preprocess-poland.log
<log_dir>/frame_analysis-finland.log
<log_dir>/laclau_discourse-portugal.log
```

There are **no per-step or per-country subdirectories**. The point is that a human troubleshooting at
02:00 lists one directory and sees every run, and that a script can find a log without knowing how the
job was launched.

## Where the directory is

`ep24_logging.log_dir()` resolves, first match wins:

| Order | Variable | Notes |
| --- | --- | --- |
| 1 | `EP24_LOG_DIR` | the canonical name |
| 2 | `LACLAUGPT_EP24_LOG_DIR` | honoured so existing SLURM jobs keep working |
| 3 | `LACLAUGPT_ANALYSIS_LOG_DIR` | honoured for the analysis-side callers |
| 4 | `<LACLAUGPT_MULTIMODAL_PRIVATE_ROOT>/logs` | default; **outside** the repository |

The directory is created with mode `0700`. Runtime logs are never committed: they belong next to the
private analysis root, not in the working tree.

## Naming rules

`<step_name>-<country>.log`, lowercase snake_case step, lowercase country:

| Step | Name |
| --- | --- |
| 0 | `import` |
| 1 | `preprocess` |
| 2 | `frame_analysis` |
| 3 | `video_analysis` |
| 4 | `summary` |
| 5 | `postprocess` |
| 6 | `laclau_discourse` |
| 7 | `discourse_network` |
| 8 | `social_network` |
| 9 | `rdf` |

Countries: `finland`, `poland`, `portugal`, plus `all` for country-agnostic steps. Aliases (`fi`, `pl`,
`pt`, `POR`) are normalized; **anything else raises instead of becoming a near-miss filename**. A typo
that silently writes `preproces-finland.log` is the failure this module exists to prevent.

## Scheduler logs

SLURM owns its own files, so they cannot share the Python log's name. They live in the **same**
directory with the job id appended:

```
<log_dir>/preprocess-poland-1234567.out
<log_dir>/preprocess-poland-1234567.err
```

`scripts/roihu/submit_step_common.sh` derives both from the step and the `--country` argument, and
exports `LACLAUGPT_STEP` / `LACLAUGPT_COUNTRY` so the Python side names its file identically.

## Rotation and reruns

A log is opened in **append** mode: a rerun never truncates the previous attempt. Before a run starts
writing, an oversized file (default 5 MB) is moved aside to
`<step>-<country>.<YYYYmmddTHHMMSS>.log`, keeping the canonical name free for the current run. Python's
`RotatingFileHandler` handles growth *during* a run, with `backupCount=5`.

Concurrent jobs for different countries never touch the same file. Two retries of the same step and
country can interleave — Linux append is atomic per write, and each record is prefixed with a timestamp
and logger name, so interleaving stays readable rather than corrupt.

## Using it in code

```python
from ep24_logging import configure, canonical_log_path

configure(step=1, country="poland")        # attaches the canonical file handler
path = canonical_log_path(1, "poland")      # where it went
```

Step modules imported before a country is known call `configure_from_env(default_step=...)`, which reads
`LACLAUGPT_STEP` and `LACLAUGPT_COUNTRY` from the submitter and falls back to `all`.

## Finding logs

```bash
export EP24_LOG_DIR=<log directory>
ls "$EP24_LOG_DIR"                      # every step and country
ls "$EP24_LOG_DIR"/preprocess-*.log      # one step across countries
tail -f "$EP24_LOG_DIR"/laclau_discourse-portugal.log
```

## What is not unified

Scheduler `--account` values and cluster paths remain cluster configuration. The `.sbatch` files no
longer hardcode a project-specific log directory; the submitter supplies `--output`/`--error` from
`EP24_LOG_DIR` at submission time.
