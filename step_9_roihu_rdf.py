#!/usr/bin/env python3
"""Step 9: deterministic RDF export after discourse/network analysis."""
import os
import sys

from ep24_cli import configure_step_cli
from roihu_csv_rdf import main


if __name__ == "__main__":
    selection = configure_step_cli(9, sys.argv[1:])
    args = list(selection.remaining_argv)
    if "--input" not in args and os.getenv("LACLAUGPT_INPUT_CSV"):
        args = ["--input", os.environ["LACLAUGPT_INPUT_CSV"], *args]
    if "--limit" not in args and selection.limit > 0:
        args += ["--limit", str(selection.limit)]
    raise SystemExit(main(args))
