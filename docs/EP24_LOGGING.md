# EP24 pipeline logs (#324)

## The convention

Every pipeline log is `<step_name>-<country>.log` in **one** directory:

```
preprocess-poland.log
frame_analysis-finland.log
laclau_discourse-portugal.log
```

- step names are lower snake_case; underscores are allowed **inside** the name
- countries are lowercase
- the separator between them is a hyphen, always
- no per-step or per-country subdirectories

## Where

`EP24_LOG_DIR` if set, otherwise `<repository>/logs`. The directory is created on demand and is
git-ignored; runtime logs are never committed.

## Scheduler output

`#SBATCH --output` points at the canonical file; `--error` points at `<step>-<country>.err.log` **in the
same directory**. stdout can share the canonical file because both the scheduler and the application
append line-oriented text. stderr cannot: an unhandled traceback interleaved with application lines is
unreadable, so it gets a sibling rather than a different directory. Use `ep24_logs.slurm_paths(step,
country)` to get both paths rather than writing them by hand.

## Concurrency and retries

`ep24_logs.start_run(step, country)` rotates the previous canonical file to
`<step>-<country>.<UTC stamp>.log` **under an exclusive lock**, then returns the canonical path, which is
opened in append mode. Two concurrent jobs therefore cannot both rotate, and neither can truncate the
other's output. A crash between rotation and open loses nothing: the rotated file is already complete.

Archives stay in the same directory, so "look in the log directory" still finds everything.

## Using it

```python
import ep24_logs

handle = ep24_logs.open_log("preprocess", "poland", run_header="run 2026-10-30")
# ... write to handle ...

# or attach a logging handler (idempotent)
ep24_logs.configure_logging("frame_analysis", "finland")
```

An unusable step or country name raises `ValueError` rather than producing a mangled path: a log called
`step--.log` is worse than a failure, because it looks like a log.

## Tests

`tests/test_ep24_logs.py` covers the filename convention, normalisation and its refusals, the configurable
root, scheduler paths, rotation, header-then-append behaviour, **two concurrent writers**, and handler
idempotence.
