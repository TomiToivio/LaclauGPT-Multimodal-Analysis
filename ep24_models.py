"""Per-stage model resolution for the EP24 Roihu pipeline (#378).

The pipeline used one shared Ollama default for every stage. That made a single environment variable
change the model for laclau discourse analysis, DNA extraction and frame analysis at once — which is
exactly the accident #378 exists to prevent: **Step 6 must stay on the 27B model.**

Resolution order per stage:

1. ``LACLAUGPT_EP24_MODEL_STEP_<n>`` — an explicit, per-stage choice.
2. ``LACLAUGPT_MULTIMODAL_MODEL`` — the shared override, for stages that are not pinned.
3. the stage default in :data:`STAGES`.

**Pinned stages ignore the shared override entirely** and say so in the log. A pinned stage is one
where a silent model change would alter research output: Step 6 (Laclau/Palonen discourse analysis).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

#: Shared override, kept for compatibility with existing runbooks.
SHARED_ENV = "LACLAUGPT_MULTIMODAL_MODEL"

#: Per-stage override pattern.
STAGE_ENV = "LACLAUGPT_EP24_MODEL_STEP_{stage}"


@dataclass(frozen=True)
class StageModel:
    """One stage's resolved model, with the evidence a startup log needs."""

    stage: int
    name: str
    backend: str
    source: str
    pinned: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {"stage": self.stage, "model": self.name, "backend": self.backend,
                "source": self.source, "pinned": self.pinned, "note": self.note}


#: Stage defaults (#378). ASR and OCR are specialist models and are not LLM stages; they are listed so
#: the table is complete and nobody wonders what Step 1 runs.
STAGES: dict[int, dict[str, object]] = {
    1: {"name": "canary-1b-v2 + PP-OCRv5", "backend": "specialist", "pinned": True,
        "note": "ASR and OCR are not language models; do not route them through Ollama or vLLM"},
    2: {"name": "qwen3-vl:8b", "backend": "ollama", "pinned": False,
        "note": "vision model; must accept image input, verify the exact tag on the host"},
    3: {"name": "Qwen/Qwen3-VL-8B-Instruct", "backend": "vllm", "pinned": False,
        "note": "native whole-video analysis; GH200 baseline"},
    4: {"name": "qwen3:8b", "backend": "ollama", "pinned": False,
        "note": "evidence-preserving summary; text only"},
    5: {"name": "qwen3:4b", "backend": "ollama", "pinned": False,
        "note": "structured postprocess; fall back to 8B if the schema gate fails"},
    6: {"name": "qwen3.8:27b", "backend": "ollama", "pinned": True,
        "note": "Laclau/Palonen discourse analysis: pinned, methodology-sensitive"},
    7: {"name": "qwen3:8b", "backend": "ollama", "pinned": False,
        "note": "DNA actor/concept/stance extraction"},
    8: {"name": "qwen3:8b", "backend": "ollama", "pinned": False,
        "note": "SNA relation extraction; descriptive only"},
    9: {"name": "none", "backend": "deterministic", "pinned": True,
        "note": "RDF projection is deterministic and must not call a model"},
}


def stage_model(stage: int) -> StageModel:
    """Resolve one stage's model, or explain which layer decided it."""
    try:
        spec = STAGES[stage]
    except KeyError:
        raise ValueError(f"unknown EP24 stage {stage}") from None

    pinned = bool(spec["pinned"])
    specific_env = STAGE_ENV.format(stage=stage)
    specific = (os.getenv(specific_env) or "").strip()
    if specific:
        return StageModel(stage, specific, str(spec["backend"]), specific_env, pinned,
                          str(spec["note"]))

    shared = (os.getenv(SHARED_ENV) or "").strip()
    if shared and not pinned:
        return StageModel(stage, shared, str(spec["backend"]), SHARED_ENV, pinned,
                          str(spec["note"]))

    note = str(spec["note"])
    if shared and pinned:
        note = (f"{note}; {SHARED_ENV} is set but ignored because this stage is pinned")
    return StageModel(stage, str(spec["name"]), str(spec["backend"]), "repository_default",
                      pinned, note)


def stage_model_source(stage: int) -> str:
    return stage_model(stage).source


def describe(stages: tuple[int, ...] = ()) -> list[dict]:
    chosen = stages or tuple(sorted(STAGES))
    return [stage_model(stage).as_dict() for stage in chosen]


def verify_available(specs: list[dict], available: set[str]) -> list[str]:
    """Which resolved Ollama models are not present on the host.

    ``--check`` exists because the issue asks for *verified, installable* tags rather than assumed
    names. A model that is not pulled is not a configuration, it is a 404 at the first call.
    """
    missing = []
    for spec in specs:
        if spec["backend"] != "ollama":
            continue
        if spec["model"] not in available:
            missing.append(f"stage {spec['stage']}: {spec['model']} (from {spec['source']})")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve the EP24 per-stage model configuration.")
    parser.add_argument("--stage", type=int, action="append", default=[],
                        help="limit to a stage (repeatable)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", metavar="AVAILABLE_TXT",
                        help="file with one available Ollama tag per line; exits 1 on a gap")
    args = parser.parse_args(argv)

    specs = describe(tuple(args.stage))
    if args.json:
        print(json.dumps(specs, indent=2))
    else:
        for spec in specs:
            flag = " (pinned)" if spec["pinned"] else ""
            print(f"step {spec['stage']}: {spec['model']}  [{spec['backend']}, via {spec['source']}]{flag}")
            if spec["note"]:
                print(f"    {spec['note']}")

    if args.check:
        try:
            available = {
                line.strip() for line in open(args.check, encoding="utf-8")
                if line.strip() and not line.startswith("#")
            }
        except OSError as exc:
            print(f"cannot read available-tags file: {exc}", file=sys.stderr)
            return 2
        missing = verify_available(specs, available)
        if missing:
            print("missing on this host:", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            return 1
        print("all resolved Ollama models are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Backwards compatibility (#156)
# --------------------------------------------------------------------------- #
# Older modules import ``ollama_model()`` with no stage. A call site that has not been given a stage
# yet must not silently inherit the pinned 27B, so it resolves through the shared override and the old
# default — the behaviour it had before this module gained a stage table.

DEFAULT_OLLAMA_MODEL = "qwen3.8:27b"
DEFAULT_VLLM_VIDEO_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def ollama_model(*, specific_env: str | None = None) -> str:
    """Legacy resolver: a specific env, then the shared override, then the historical default."""
    if specific_env:
        value = os.getenv(specific_env)
        if value:
            return value
    return os.getenv(SHARED_ENV, DEFAULT_OLLAMA_MODEL)


def ollama_model_source(*, specific_env: str | None = None) -> str:
    if specific_env and os.getenv(specific_env):
        return specific_env
    if os.getenv(SHARED_ENV):
        return SHARED_ENV
    return "repository_default"
