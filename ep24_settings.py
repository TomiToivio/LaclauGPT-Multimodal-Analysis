"""Load EP24 private runtime settings without exposing or logging secrets."""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_PRIVATE_ROOT = Path(
    "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess"
)


def private_root() -> Path:
    return Path(
        os.getenv("LACLAUGPT_EP24_PRIVATE_ROOT")
        or os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT")
        or _DEFAULT_PRIVATE_ROOT
    )


def load_private_env(path: str | Path | None = None) -> Path:
    """Load private settings, then derive every EP24 path from the resolved root."""
    if path is not None:
        env_path = Path(path)
    elif os.getenv("LACLAUGPT_EP24_ENV_FILE"):
        env_path = Path(os.environ["LACLAUGPT_EP24_ENV_FILE"])
    else:
        env_path = private_root() / ".env"

    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)

    # The .env may relocate the private root. Resolve it only after parsing the
    # file, then keep the two historical aliases coherent for all callers.
    resolved_root = private_root()
    os.environ.setdefault("LACLAUGPT_EP24_PRIVATE_ROOT", str(resolved_root))
    os.environ.setdefault("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", str(resolved_root))
    os.environ.setdefault("LACLAUGPT_MONGO_ENABLED", "1")
    os.environ.setdefault("LACLAUGPT_DATASET", "ep2024_reprocess")
    os.environ.setdefault(
        "LACLAUGPT_EP24_INPUT_ROOT",
        str(resolved_root / "data" / "to_reprocess"),
    )
    os.environ.setdefault(
        "LACLAUGPT_EP24_OUTPUT_ROOT",
        str(resolved_root / "outputs"),
    )
    return env_path

# --------------------------------------------------------------------------- #
# Validation and reporting (issue #188)
# --------------------------------------------------------------------------- #
#
# The helpers above mutate os.environ and are what the pipeline steps use. The
# functions below exist so the Roihu bootstrap can *report* what resolved and
# name a missing setting instead of failing with a stack trace, without ever
# printing a value.

#: Setting name fragments that mark a value as secret. Anything matching is
#: reported by name and length, never by value.
SECRET_MARKERS = (
    "URI", "URL", "KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL",
    "CONNECTION", "DSN", "PRIVATE",
)

#: Settings the Roihu steps 1-6 need. A missing one is reported, not raised, so
#: the bootstrap can list them all in one run.
REQUIRED_SETTINGS: tuple[str, ...] = ("LACLAUGPT_MONGO_URI",)


def is_secret(name: str) -> bool:
    """Whether a setting name looks like it carries a credential."""
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def redact(value: str) -> str:
    """A safe rendering of a value for logs: never the value itself."""
    if not value:
        return "<empty>"
    return f"<redacted {len(value)} chars>"


def describe(environ: dict[str, str] | None = None) -> list[str]:
    """One safe line per set LACLAUGPT_* setting: name, and redacted-or-plain value."""
    env = environ if environ is not None else dict(os.environ)
    lines: list[str] = []
    for name in sorted(env):
        if not name.startswith("LACLAUGPT_"):
            continue
        value = env[name]
        if not value:
            continue
        lines.append(f"{name}={redact(value) if is_secret(name) else value}")
    return lines


def validate(
    environ: dict[str, str] | None = None,
    required: tuple[str, ...] = REQUIRED_SETTINGS,
) -> list[str]:
    """Return the names of required settings that are unset or empty."""
    env = environ if environ is not None else dict(os.environ)
    if env.get("LACLAUGPT_MONGO_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return []
    return [name for name in required if not env.get(name)]


def summary(environ: dict[str, str] | None = None) -> str:
    """A multi-line, secret-free report for bootstrap logs."""
    env = environ if environ is not None else dict(os.environ)
    lines = [f"private_root={private_root()}"]
    env_file = env.get("LACLAUGPT_EP24_ENV_FILE") or str(private_root() / ".env")
    lines.append(f"env_file={env_file}{'' if Path(env_file).exists() else ' (not found)'}")
    lines.extend(f"  {line}" for line in describe(env))
    missing = validate(env)
    lines.append(f"missing_required={missing if missing else 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI for the Roihu bootstrap: report settings, redacting every secret."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Report resolved EP24 private settings. Values of anything that "
                    "looks like a credential are redacted, so this output is safe to paste."
    )
    parser.add_argument("--private-root", default=None,
                        help="private checkout to read .env from (sets LACLAUGPT_EP24_PRIVATE_ROOT)")
    parser.add_argument("--env-file", default=None, help="explicit settings file")
    args = parser.parse_args(argv)

    if args.private_root:
        os.environ["LACLAUGPT_EP24_PRIVATE_ROOT"] = args.private_root
    load_private_env(args.env_file)
    print(summary())
    missing = validate()
    if missing:
        print(
            f"WARNING: {len(missing)} required setting(s) missing: {', '.join(missing)}. "
            "Set them in the private .env (see docs/ROIHU_RUNBOOK.md).",
            file=__import__("sys").stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
