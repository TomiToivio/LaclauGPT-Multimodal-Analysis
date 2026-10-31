#!/usr/bin/env python3
"""Propose codebook merges without changing anything (issue #333).

Reads one private EP24 country codebook and reports groups that are plausibly the same entry:
labels that are identical after case/whitespace/punctuation normalisation, and entries that share an
alias. It never edits the codebook and never resolves ambiguity; a researcher reviews every group.

    python scripts/ep24/propose_codebook_merges.py --codebook <private>/ep24_finland_private.json
    python scripts/ep24/propose_codebook_merges.py --codebook ... --json

The output is local: labels stay in the private checkout and are not written to the repository.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalise(label: str) -> str:
    text = unicodedata.normalize("NFKC", label).casefold().strip()
    text = re.sub(r"^[#@]+", "", text)
    text = re.sub(r"[\W_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entries_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("entries")
    if not isinstance(rows, list):
        raise ValueError("codebook has no 'entries' list")
    return [row for row in rows if isinstance(row, dict)]


def proposals(document: dict[str, Any]) -> list[dict[str, Any]]:
    entries = entries_of(document)
    by_norm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_alias: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        kind = str(entry.get("kind", ""))
        label = str(entry.get("label", ""))
        if label:
            by_norm[(kind, normalise(label))].append(entry)
        for alias in entry.get("aliases", []) or []:
            alias_norm = normalise(str(alias))
            if alias_norm:
                by_alias[(kind, alias_norm)].append(entry)

    found: list[dict[str, Any]] = []
    for (kind, norm), group in sorted(by_norm.items()):
        if len(group) > 1:
            found.append({
                "reason": "same_normalised_label",
                "kind": kind,
                "normalised": norm,
                "entries": [str(entry.get("label", "")) for entry in group],
            })
    for (kind, alias), group in sorted(by_alias.items()):
        unique = {id(entry): entry for entry in group}
        if len(unique) > 1:
            found.append({
                "reason": "shared_alias",
                "kind": kind,
                "alias": alias,
                "entries": [str(entry.get("label", "")) for entry in unique.values()],
            })
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.codebook.read_text(encoding="utf-8"))
        found = proposals(document)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot audit {args.codebook}: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"proposals": found, "count": len(found)}, ensure_ascii=False, indent=2))
    else:
        print(f"{len(found)} proposal(s) in {args.codebook.name}; nothing was modified")
        for item in found:
            print(f"  {item['reason']} ({item['kind']}): {len(item['entries'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
