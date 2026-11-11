"""Shared active-model defaults for the EP24 Roihu pipeline."""
from __future__ import annotations

import os

DEFAULT_OLLAMA_MODEL = "qwen3.8:27b"
DEFAULT_VLLM_VIDEO_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


STAGE_MODELS = {
    2: "qwen3-vl:8b-instruct-q4_K_M",
    4: "qwen3:8b-q4_K_M",
    5: "qwen3:4b-q4_K_M",
    6: DEFAULT_OLLAMA_MODEL,
    7: "qwen3:8b-q4_K_M",
    8: "qwen3:4b-q4_K_M",
}


def _stage_selection(stage: int) -> tuple[str, str]:
    if stage not in STAGE_MODELS:
        raise ValueError(f"Step {stage} is not an Ollama stage")
    key = f"LACLAUGPT_STEP_{stage}_MODEL"
    if stage == 6:
        if os.getenv(key, DEFAULT_OLLAMA_MODEL) != DEFAULT_OLLAMA_MODEL:
            raise ValueError("Step 6 must remain pinned to qwen3.8:27b")
        return DEFAULT_OLLAMA_MODEL, "step_6_pinned"
    profile = os.getenv("LACLAUGPT_MODEL_PROFILE", "preserve")
    if profile not in {"preserve", "small-candidate"}:
        raise ValueError(f"Unknown model profile: {profile}")
    if os.getenv(key):
        return os.environ[key], key
    if profile == "small-candidate":
        return STAGE_MODELS[stage], "small-candidate"
    return ollama_model(), ollama_model_source()


def ollama_model(*, specific_env: str | None = None, stage: int | None = None) -> str:
    """Resolve a step-specific override, then the shared override, then default."""
    if stage is not None:
        return _stage_selection(stage)[0]
    if specific_env:
        value = os.getenv(specific_env)
        if value:
            return value
    return os.getenv("LACLAUGPT_MULTIMODAL_MODEL", DEFAULT_OLLAMA_MODEL)


def ollama_model_source(*, specific_env: str | None = None, stage: int | None = None) -> str:
    """Human-readable source of the selected Ollama model for startup logs."""
    if stage is not None:
        return _stage_selection(stage)[1]
    if specific_env and os.getenv(specific_env):
        return specific_env
    if os.getenv("LACLAUGPT_MULTIMODAL_MODEL"):
        return "LACLAUGPT_MULTIMODAL_MODEL"
    return "repository_default"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve the same model as the stage runner")
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGE_MODELS))
    args = parser.parse_args()
    print(ollama_model(stage=args.stage))
