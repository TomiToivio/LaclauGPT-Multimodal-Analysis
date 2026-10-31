"""One log directory, one filename convention, for every EP24 pipeline step (#324).

Before this module, a step could write a log relative to its working directory, a launcher could send
stdout somewhere else, and a SLURM directive could name a third path. Finding "the log for Poland step 2"
meant knowing which of those produced it.

The convention here is ``<step_name>-<country>.log`` in a single configurable directory:

* step names are lower snake_case (underscores allowed *inside* the name),
* countries are lowercase,
* the separator between them is a hyphen, always,

so ``frame_analysis`` + ``Finland`` becomes ``frame_analysis-finland.log``. Names are normalised
centrally instead of at each call site, and an unusable name raises rather than silently producing a
mangled path.

Concurrency: two jobs for the same step and country must not truncate each other. :func:`start_run`
takes an exclusive lock, rotates the previous canonical file to a timestamped sibling in the same
directory (never a subdirectory), then the caller opens the canonical path in append mode. A crash
between rotation and open loses nothing: the rotated file is already complete.
"""
from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ENV_LOG_DIR = "EP24_LOG_DIR"
_CANONICAL_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def log_dir(create: bool = True) -> Path:
    """The one log directory. ``EP24_LOG_DIR`` wins; the default is ``<repo>/logs``."""
    configured = os.environ.get(ENV_LOG_DIR)
    if configured:
        path = Path(configured).expanduser()
    else:
        path = Path(__file__).resolve().parent / "logs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def normalise_step(step: str) -> str:
    """Lower snake_case, with the same explicit refusals as :func:`normalise_country`."""
    if step is None:
        raise ValueError("step name is required")
    cleaned = str(step).strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"step name {step!r} normalises to nothing")
    if not _CANONICAL_NAME.match(cleaned):
        raise ValueError(
            f"step name {step!r} normalises to {cleaned!r}, which is not lower snake_case"
        )
    return cleaned


def normalise_country(country: str) -> str:
    """Lowercase, single token. 'FI' and ' Finland ' both become 'finland'."""
    if country is None:
        raise ValueError("country is required")
    cleaned = str(country).strip().lower().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"country {country!r} normalises to nothing")
    if not _CANONICAL_NAME.match(cleaned):
        raise ValueError(f"country {country!r} normalises to {cleaned!r}, which is not a single token")
    return cleaned


def log_path(step: str, country: str) -> Path:
    """``<step>-<country>.log`` — the canonical, discoverable name."""
    return log_dir() / f"{normalise_step(step)}-{normalise_country(country)}.log"


def error_path(step: str, country: str) -> Path:
    """Scheduler stderr for the same run, in the same directory and trivially associated.

    stdout can share the canonical file safely because both the application and the scheduler append
    line-oriented text. stderr cannot: an unhandled traceback written concurrently with application
    lines produces unreadable interleaving, so it gets its own ``.err.log`` sibling rather than a
    different directory.
    """
    return log_dir() / f"{normalise_step(step)}-{normalise_country(country)}.err.log"


def slurm_paths(step: str, country: str) -> dict[str, str]:
    """Values for ``#SBATCH --output=`` and ``--error=``, kept in the configured directory."""
    return {"output": str(log_path(step, country)), "error": str(error_path(step, country))}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@contextlib.contextmanager
def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def start_run(step: str, country: str) -> Path:
    """Rotate any previous canonical log, then return the canonical path for this run.

    Rotation only happens under an exclusive lock, so two concurrent starts cannot both rotate and one
    cannot delete the other's work. The rotated file keeps the canonical stem plus a UTC stamp, in the
    same directory, so discovery still means "look in the log directory".
    """
    canonical = log_path(step, country)
    lock = canonical.with_suffix(".lock")
    with _locked(lock):
        if canonical.exists() and canonical.stat().st_size > 0:
            archive = canonical.with_name(f"{canonical.stem}.{_stamp()}.log")
            canonical.rename(archive)
    canonical.touch(exist_ok=True)
    return canonical


def open_log(step: str, country: str, run_header: str | None = None):
    """Open the canonical log for appending, writing a run header when the file is new.

    Opening deliberately does **not** rotate. Rotation is a run-level decision and belongs to
    :func:`start_run`; making open rotate would let a second handle in the same process destroy the
    first one's output.
    """
    path = log_path(step, country)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    if path.stat().st_size == 0:
        header = run_header or f"run start {_stamp()} pid={os.getpid()}"
        handle.write(f"# {header}\n")
        handle.flush()
    return handle


def configure_logging(step: str, country: str, level: int = logging.INFO, logger_name: str = "ep24") -> Path:
    """Attach a file handler for this step/country to ``logger_name``, idempotently.

    Calling twice for the same path does not double every line, which is the classic way a "just add a
    handler" change turns logs into noise.
    """
    path = log_path(step, country)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    target = str(path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and str(Path(handler.baseFilename).resolve()) == target:
            return path
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return path
