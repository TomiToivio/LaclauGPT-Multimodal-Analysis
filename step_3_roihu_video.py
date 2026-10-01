#!/usr/bin/env python3
"""Step 3: native whole-video Qwen3-VL/vLLM analysis on CSC Roihu.

Consumes the complete Step 2 dataframe. Every incoming field, including source
metadata, Whisper/translation/OCR and the deep t=1.0s frame analysis, is supplied
as cumulative prompt context and preserved in the output CSV.

The whole-video stage complements the still-frame stage by describing temporal
narrative, ordered events, actions, scene changes and additional feed-scroll
failures. The shared EP24 rule removes the first 1.0s non-destructively.

Generic pipeline env vars are bridged to the tested vLLM harness:
- LACLAUGPT_INPUT_CSV  -> LACLAUGPT_VLLM_TEST_INPUT_CSV
- LACLAUGPT_OUTPUT_CSV -> LACLAUGPT_VLLM_TEST_OUTPUT_CSV

All rows are processed by default. LACLAUGPT_MAX_ROWS is an optional test limit.
"""
import os

from experiments.vllm_video_test import main


def _bridge_env() -> None:
    # Production orchestration is authoritative. Explicit cumulative stage paths
    # must override stale experiment variables inherited from the private .env.
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        os.environ["LACLAUGPT_VLLM_TEST_INPUT_CSV"] = os.environ["LACLAUGPT_INPUT_CSV"]
    if os.getenv("LACLAUGPT_OUTPUT_CSV"):
        os.environ["LACLAUGPT_VLLM_TEST_OUTPUT_CSV"] = os.environ["LACLAUGPT_OUTPUT_CSV"]


if __name__ == "__main__":
    _bridge_env()
    argv: list[str] = []
    limit = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if limit > 0:
        argv.extend(["--sample-size", str(limit)])
    raise SystemExit(main(argv))
