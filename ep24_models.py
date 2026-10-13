"""Shared active-model defaults for the EP24 Roihu pipeline."""
from __future__ import annotations

import os

DEFAULT_OLLAMA_MODEL = "qwen3.8:27b"
DEFAULT_VLLM_VIDEO_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def ollama_model(*, specific_env: str | None = None) -> str:
    """Resolve a step-specific override, then the shared override, then default."""
    if specific_env:
        value = os.getenv(specific_env)
        if value:
            return value
    return os.getenv("LACLAUGPT_MULTIMODAL_MODEL", DEFAULT_OLLAMA_MODEL)


def ollama_model_source(*, specific_env: str | None = None) -> str:
    """Human-readable source of the selected Ollama model for startup logs."""
    if specific_env and os.getenv(specific_env):
        return specific_env
    if os.getenv("LACLAUGPT_MULTIMODAL_MODEL"):
        return "LACLAUGPT_MULTIMODAL_MODEL"
    return "repository_default"
