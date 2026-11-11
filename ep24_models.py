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


STAGE_MODELS = {
    2: "qwen3-vl:8b-instruct-q4_K_M",
    4: "qwen3:8b-q4_K_M",
    5: "qwen3:4b-q4_K_M",
    6: DEFAULT_OLLAMA_MODEL,
    7: "qwen3:8b-q4_K_M",
    8: "qwen3:4b-q4_K_M",
}


def model_profile() -> str:
    profile = os.getenv("LACLAUGPT_MODEL_PROFILE", "preserve")
    if profile not in {"preserve", "small-candidate"}:
        raise ValueError(f"Unknown model profile: {profile}")
    return profile


def stage_ollama_model(step: int) -> str:
    """Resolve a pinned per-stage model without leaking shared defaults into Step 6.

    Model migration is opt-in for other stages until validation of exact model
    tags/quantization on Roihu. Step 6 retains its independent 27B baseline.
    """
    if step not in (2, 4, 5, 6, 7, 8):
        raise ValueError(f"Unsupported Ollama model stage: {step}")
    key = f"LACLAUGPT_STEP{step}_MODEL"
    if step == 6:
        configured = os.getenv(key) or DEFAULT_OLLAMA_MODEL
        if configured != DEFAULT_OLLAMA_MODEL:
            raise ValueError(
                f"Step 6 is pinned to {DEFAULT_OLLAMA_MODEL}; refusing {configured!r}"
            )
        return configured
    if model_profile() == "small-candidate" and not os.getenv(key):
        return STAGE_MODELS[step]
    return ollama_model(specific_env=key)


def stage_ollama_model_source(step: int) -> str:
    """Describe which variable selected a stage's model."""
    key = f"LACLAUGPT_STEP{step}_MODEL"
    if os.getenv(key):
        return key
    if step == 6:
        return "step6_pinned_27b_default"
    if model_profile() == "small-candidate":
        return "small-candidate"
    return ollama_model_source(specific_env=key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve the same model as the stage runner")
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGE_MODELS))
    args = parser.parse_args()
    print(stage_ollama_model(args.stage))
