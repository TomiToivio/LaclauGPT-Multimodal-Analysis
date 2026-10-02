#!/usr/bin/env python3
"""Step 3: native whole-video Qwen3-VL/vLLM analysis on CSC Roihu.

Consumes the complete Step 2 dataframe. LACLAUGPT_INPUT_CSV and
LACLAUGPT_OUTPUT_CSV remain the cumulative stage bridge. LACLAUGPT_MAX_ROWS
remains the environment-compatible optional test limit; without it or --limit,
the production wrapper processes all rows.
"""
import os
import sys

from ep24_cli import configure_step_cli
from experiments.vllm_video_test import main


def _bridge_env() -> None:
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        os.environ["LACLAUGPT_VLLM_TEST_INPUT_CSV"] = os.environ["LACLAUGPT_INPUT_CSV"]
    if os.getenv("LACLAUGPT_OUTPUT_CSV"):
        os.environ["LACLAUGPT_VLLM_TEST_OUTPUT_CSV"] = os.environ["LACLAUGPT_OUTPUT_CSV"]


if __name__ == "__main__":
    selection = configure_step_cli(3, sys.argv[1:])
    _bridge_env()
    argv = list(selection.remaining_argv)
    limit = selection.limit
    if limit > 0 and "--sample-size" not in argv:
        argv.extend(["--sample-size", str(limit)])
    raise SystemExit(main(argv))
