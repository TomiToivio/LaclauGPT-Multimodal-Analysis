"""Shared command-line contract for numbered EP24 Roihu pipeline steps.

The numbered step scripts intentionally expose only a tiny common surface:
--country/-c and --limit/-n.  Explicit CLI values override environment values;
environment values override each step's historical defaults.

When a country is selected for a direct numbered-step invocation and explicit
LACLAUGPT_INPUT_CSV/LACLAUGPT_OUTPUT_CSV paths are not already supplied, this
module resolves the deterministic cumulative checkpoint paths used by the
restartable pipeline.  This makes small Finland/Poland/Portugal samples flow
through the same rows at every stage.
"""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("ep24_cli")

DEFAULT_OLLAMA_MODEL = "qwen3.8:27b"

COUNTRY_ALIASES = {
    "finland": "finland", "fi": "finland",
    "poland": "poland", "pl": "poland",
    "portugal": "portugal", "pt": "portugal",
    "germany": "germany", "de": "germany",
    "spain": "spain", "es": "spain",
    "hungary": "hungary", "hu": "hungary",
    "croatia": "croatia", "hr": "croatia",
    "france": "france", "fr": "france",
    "bulgaria": "bulgaria", "bg": "bulgaria",
    "sweden": "sweden", "sv": "sweden",
}

COUNTRY_TO_LANGUAGE = {
    "finland": "fi",
    "poland": "pl",
    "portugal": "pt",
    "germany": "de",
    "spain": "es",
    "hungary": "hu",
    "croatia": "hr",
    "france": "fr",
    "bulgaria": "bg",
    "sweden": "sv",
}

STAGE_NAMES = {
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


@dataclass(frozen=True)
class StepSelection:
    country: str | None
    limit: int
    country_source: str
    limit_source: str
    remaining_argv: list[str]


def resolve_model(
    *variables: str,
    default: str = DEFAULT_OLLAMA_MODEL,
    logger: logging.Logger | None = None,
) -> str:
    """Resolve an inference model from an override chain and log where it came from.

    ``variables`` are the override environment variables in precedence order,
    most specific first. The first non-empty value wins; otherwise ``default``.

    Issue #156 requirement 5 asks each model-using step to log the selected model
    and whether it came from an explicit override or the repository default. Doing
    that in one place keeps the resolution hierarchy identical across steps.
    Only the variable name and the model identifier are logged, never values of
    anything else.
    """
    out = logger or LOG
    for name in variables:
        value = str(os.getenv(name) or "").strip()
        if value:
            out.info("model=%s source=%s", value, name)
            return value
    out.info("model=%s source=repository-default", default)
    return default


def normalize_country(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    key = str(value).strip().casefold()
    try:
        return COUNTRY_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(COUNTRY_ALIASES.values())))
        raise ValueError(f"unknown country {value!r}; choose one of: {allowed}") from exc


def _positive_limit(value: str | int | None, *, source: str) -> int:
    if value is None or str(value).strip() == "":
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{source} must be >= 0")
    return parsed


def checkpoint_path(step: int, country: str, output_root: Path) -> Path:
    name = STAGE_NAMES[step]
    return output_root / country / f"step_{step:02d}_{name}.csv"


def _configure_direct_paths(step: int, country: str) -> None:
    """Fill direct-run paths only when the caller did not provide explicit ones."""
    input_root = Path(os.getenv(
        "LACLAUGPT_EP24_INPUT_ROOT",
        "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess",
    ))
    output_root = Path(os.getenv(
        "LACLAUGPT_EP24_OUTPUT_ROOT",
        "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs",
    ))

    if not os.getenv("LACLAUGPT_INPUT_CSV"):
        if step == 1:
            os.environ["LACLAUGPT_INPUT_CSV"] = str(input_root / f"ep24_{country}.csv")
        else:
            os.environ["LACLAUGPT_INPUT_CSV"] = str(checkpoint_path(step - 1, country, output_root))

    if not os.getenv("LACLAUGPT_OUTPUT_CSV"):
        os.environ["LACLAUGPT_OUTPUT_CSV"] = str(checkpoint_path(step, country, output_root))


def configure_step_cli(
    step: int,
    argv: Sequence[str] | None = None,
    *,
    configure_paths: bool = True,
) -> StepSelection:
    if step not in STAGE_NAMES:
        raise ValueError(f"invalid pipeline step: {step}")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-c", "--country", help="Process one EP24 country (e.g. finland, poland, portugal)")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        help="Maximum records/videos to process; 0 means no explicit limit",
    )
    args, remaining = parser.parse_known_args(list(argv) if argv is not None else None)

    env_country_raw = os.getenv("LACLAUGPT_COUNTRY")
    if args.country is not None:
        country = normalize_country(args.country)
        country_source = "cli"
    elif env_country_raw:
        country = normalize_country(env_country_raw)
        country_source = "environment"
    else:
        country = None
        country_source = "default"

    if args.limit is not None:
        limit = _positive_limit(args.limit, source="--limit")
        limit_source = "cli"
    elif os.getenv("LACLAUGPT_MAX_ROWS") not in (None, ""):
        limit = _positive_limit(os.getenv("LACLAUGPT_MAX_ROWS"), source="LACLAUGPT_MAX_ROWS")
        limit_source = "environment"
    else:
        limit = 0
        limit_source = "default"

    if country is not None:
        os.environ["LACLAUGPT_COUNTRY"] = country
        os.environ["LACLAUGPT_LANGUAGES"] = COUNTRY_TO_LANGUAGE[country]
        if configure_paths:
            _configure_direct_paths(step, country)
    if limit_source != "default":
        os.environ["LACLAUGPT_MAX_ROWS"] = str(limit)

    message = (
        f"runtime_selection step={step} country={country or '<default>'} limit={limit} "
        f"selection_source.country={country_source} selection_source.limit={limit_source} "
        f"input={os.getenv('LACLAUGPT_INPUT_CSV', '<default>')} "
        f"output={os.getenv('LACLAUGPT_OUTPUT_CSV', '<default>')}"
    )
    LOG.info(message)
    # The numbered entry points configure logging at different times. Emit one
    # concise startup line regardless, so sbatch/cron logs always show selection.
    print(message, file=__import__("sys").stderr)
    return StepSelection(country, limit, country_source, limit_source, remaining)
