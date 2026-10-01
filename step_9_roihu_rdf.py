#!/usr/bin/env python3
"""Step 9: deterministic RDF export after discourse/network analysis."""
import os
import sys
from roihu_csv_rdf import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args and os.getenv("LACLAUGPT_INPUT_CSV"):
        args = ["--input", os.environ["LACLAUGPT_INPUT_CSV"]]
    limit = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if "--limit" not in args and limit > 0:
        args += ["--limit", str(limit)]
    raise SystemExit(main(args))
