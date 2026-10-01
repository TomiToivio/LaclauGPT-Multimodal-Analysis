"""Load EP24 private runtime settings without exposing or logging secrets."""
from __future__ import annotations

import os
from pathlib import Path


def private_root() -> Path:
    return Path(
        os.getenv("LACLAUGPT_EP24_PRIVATE_ROOT")
        or os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT")
        or "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess"
    )


def load_private_env(path: str | Path | None = None) -> Path:
    env_path = Path(path) if path else Path(
        os.getenv("LACLAUGPT_EP24_ENV_FILE") or private_root() / ".env"
    )
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

    os.environ.setdefault("LACLAUGPT_MONGO_ENABLED", "1")
    os.environ.setdefault("LACLAUGPT_DATASET", "ep2024_reprocess")
    os.environ.setdefault("LACLAUGPT_EP24_PRIVATE_ROOT", str(private_root()))
    os.environ.setdefault(
        "LACLAUGPT_EP24_INPUT_ROOT",
        str(private_root() / "data" / "to_reprocess"),
    )
    os.environ.setdefault(
        "LACLAUGPT_EP24_OUTPUT_ROOT",
        str(private_root() / "outputs"),
    )
    return env_path
