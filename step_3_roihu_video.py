#!/usr/bin/env python3
"""Step 3 (optional): native whole-video VLM analysis.

This stage is reserved now so later steps keep stable numbers. It delegates to
the isolated vLLM experiment until that backend is accepted as production-ready.
Set the input/output paths with the experiment's documented environment
variables. The first second is removed by the shared EP24 media contract.
"""
import os
from experiments.vllm_video_test import main

if __name__ == "__main__":
    limit = os.getenv("LACLAUGPT_MAX_ROWS", "100")
    raise SystemExit(main(["--sample-size", limit]))
