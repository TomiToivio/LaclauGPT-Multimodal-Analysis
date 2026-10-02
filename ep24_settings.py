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
