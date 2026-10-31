"""One log directory, one filename convention for every EP24 pipeline step (#324).

Every step and country writes to ``<log_dir>/<step_name>-<country>.log``:

    preprocess-poland.log
    frame_analysis-finland.log
    laclau_discourse-portugal.log

The directory is configured once, by ``EP24_LOG_DIR`` (with the existing
``LACLAUGPT_EP24_LOG_DIR`` / ``LACLAUGPT_ANALYSIS_LOG_DIR`` honoured as fallbacks so current SLURM and
analysis jobs keep working). There are **no per-step or per-country subdirectories** — the point of the
issue is that a human can list one directory and see every run.

Names are derived here and nowhere else. A step or country that cannot be normalized raises instead of
producing a near-miss filename, because a typo that silently writes ``preproces-finland.log`` is exactly
the failure this module exists to prevent.

Concurrency: SLURM runs one job per country, but retries can overlap. Log files are opened in append
mode and, when a file grows past ``MAX_BYTES``, rotated to ``<name>.<timestamp>.log`` before the new run
starts writing. The canonical filename always exists and never loses the earlier content.
"""
from __future__ import annotations

import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = [
    "COUNTRY_ALIASES",
    "MAX_BYTES",
    "STEP_NAMES",
    "canonical_log_path",
    "configure",
    "configure_from_env",
    "log_dir",
    "normalize_country",
    "normalize_step",
    "slurm_log_path",
]

#: Primary environment variable. The two legacy names are still honoured.
ROOT_ENV_VARS = ("EP24_LOG_DIR", "LACLAUGPT_EP24_LOG_DIR", "LACLAUGPT_ANALYSIS_LOG_DIR")

#: Canonical step names. Keys are the pipeline step numbers used by the step_* entrypoints.
STEP_NAMES: dict[int, str] = {
    0: "import",
    1: "preprocess",
    2: "frame_analysis",
    3: "video_analysis",
    4: "summary",
    5: "postprocess",
    6: "laclau_discourse",
    7: "discourse_network",
    8: "social_network",
    9: "rdf",
}

#: Extra spellings that may arrive from the CLI, SLURM or older configs.
STEP_ALIASES: dict[str, int] = {
    "video": 3,
    "discourse": 6,
    "sna": 8,
    "network": 8,
}

COUNTRY_ALIASES: dict[str, str] = {
    "fi": "finland",
    "fin": "finland",
    "finland": "finland",
    "pl": "poland",
    "pol": "poland",
    "poland": "poland",
    "pt": "portugal",
    "port": "portugal",
    "portugal": "portugal",
    # Country-agnostic steps (import, rdf) still need a filename; "all" is explicit rather than a guess.
    "all": "all",
}

MAX_BYTES = 5 * 1024 * 1024

_SNAKE = re.compile(r"[^a-z0-9]+")


def normalize_step(step: int | str) -> str:
    """Canonical step name. Accepts a step number, a canonical name, or a known alias."""
    if isinstance(step, bool):  # bool is an int; reject it explicitly rather than mapping True to 1
        raise ValueError(f"invalid step: {step!r}")
    if isinstance(step, int):
        if step not in STEP_NAMES:
            raise ValueError(f"unknown pipeline step {step}; known steps are {sorted(STEP_NAMES)}")
        return STEP_NAMES[step]
    cleaned = _SNAKE.sub("_", str(step).strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("step name is empty")
    if cleaned in STEP_NAMES.values():
        return cleaned
    if cleaned in STEP_ALIASES:
        return STEP_NAMES[STEP_ALIASES[cleaned]]
    if cleaned.isdigit():
        return normalize_step(int(cleaned))
    raise ValueError(f"unknown step {step!r}; known steps are {sorted(STEP_NAMES.values())}")


def normalize_country(country: str | None) -> str:
    """Canonical country name. Empty or unknown values raise; they never become a filename."""
    if country is None:
        raise ValueError("country is required")
    cleaned = _SNAKE.sub("_", str(country).strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("country is empty")
    if cleaned not in COUNTRY_ALIASES:
        raise ValueError(f"unknown country {country!r}; known countries are "
                         f"{sorted(set(COUNTRY_ALIASES.values()))}")
    return COUNTRY_ALIASES[cleaned]


def log_dir(root: str | Path | None = None) -> Path:
    """The single log directory, created if needed.

    Resolution order: explicit ``root`` argument, ``EP24_LOG_DIR``, ``LACLAUGPT_EP24_LOG_DIR``,
    ``LACLAUGPT_ANALYSIS_LOG_DIR``, then ``<LACLAUGPT_MULTIMODAL_PRIVATE_ROOT>/logs``. Runtime logs are
    never written into the repository working tree by default.
    """
    if root is not None:
        path = Path(root)
    else:
        chosen = next((os.environ[name] for name in ROOT_ENV_VARS if os.environ.get(name)), None)
        if chosen:
            path = Path(chosen)
        else:
            private_root = Path(os.environ.get("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "logs"))
            path = private_root / "logs" if private_root.name != "logs" else private_root
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def canonical_log_path(step: int | str, country: str, root: str | Path | None = None) -> Path:
    """``<log_dir>/<step_name>-<country>.log``."""
    return log_dir(root) / f"{normalize_step(step)}-{normalize_country(country)}.log"


def slurm_log_path(step: int | str, country: str | None, job_id: str,
                   stream: str = "out", root: str | Path | None = None) -> Path:
    """Scheduler stdout/stderr for a job, in the same directory but never the same file as Python logs.

    SLURM owns these files and appends ``%j`` itself, so they cannot share the canonical name; keeping
    them beside it is what makes one directory sufficient for troubleshooting.
    """
    if stream not in ("out", "err"):
        raise ValueError("stream must be 'out' or 'err'")
    name = normalize_step(step)
    suffix = f"-{normalize_country(country)}" if country else ""
    return log_dir(root) / f"{name}{suffix}-{job_id}.{stream}"


def _rotate_if_large(path: Path, max_bytes: int = MAX_BYTES) -> Path | None:
    """Move an oversized canonical log aside, keeping its content and the canonical name free."""
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return None
        archived = path.with_name(f"{path.stem}.{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}")
        path.rename(archived)
        return archived
    except OSError:
        return None


def configure_from_env(default_step: int | str | None = None,
                       level: int = logging.INFO) -> Path:
    """Configure from ``LACLAUGPT_STEP`` / ``LACLAUGPT_COUNTRY``, which the SLURM submitter exports.

    Step modules imported before a country is known fall back to ``all`` — a real filename with an
    honest meaning, rather than a per-module ``./logs/frame.log`` outside the log directory.
    """
    step: int | str = os.environ.get("LACLAUGPT_STEP") or default_step or "all"
    if isinstance(step, str) and step.lower() == "all":
        raise ValueError("configure_from_env needs a step number or name")
    country = os.environ.get("LACLAUGPT_COUNTRY") or "all"
    return configure(step, country, level=level)


def configure(step: int | str, country: str, level: int = logging.INFO,
              root: str | Path | None = None, max_bytes: int = MAX_BYTES) -> Path:
    """Attach a rotating canonical file handler to the root logger and return the path."""
    path = canonical_log_path(step, country, root)
    _rotate_if_large(path, max_bytes)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=5, encoding="utf-8", delay=False)
    os.chmod(path, 0o600) if path.exists() else None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    return path
