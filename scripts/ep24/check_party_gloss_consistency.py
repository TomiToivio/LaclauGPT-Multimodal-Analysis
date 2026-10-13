#!/usr/bin/env python3
"""Detect contradictory party glosses in an EP24 country codebook (issue #91).

The defect this catches
-----------------------
A codebook label of the form `NativeName (Gloss)` asserts that the gloss is that
entity's name in another language. When the gloss belongs to a *different*
entity, the label is not merely untidy — it is a false identity, and because the
label is a retrieval match target, a query about one party can surface the other.

Found in the Sweden pass: `Socialdemokraterna (Sweden Democrats)`. The Social
Democrats (S) and the Sweden Democrats (SD) are different parties. A query for
"Sweden Democrats" returns the Social Democrats entry.

What this checks
----------------
It does NOT try to resolve entities or guess glosses — that would be inventing
data. It reports the *contradiction*: a label whose native part contains the same
stem as a second party AND whose gloss contains the other party's name. Those
need a human decision, so they are reported, never rewritten.

Read-only. Exit 0 = no contradictions found, 1 = contradictions found.

Usage:
    python3 scripts/ep24/check_party_gloss_consistency.py --codebook <book.json> \
        --parties <pairs.json> [--json]

The parties file maps a native stem to its correct English name(s), e.g.

    {"Socialdemokraterna": ["Social Democratic Party"],
     "Sverigedemokraterna": ["Sweden Democrats"]}

It must come from the country's own public sources; agents should not invent it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


def fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def label_of(entry: dict[str, Any]) -> str:
    for key in ("label", "name", "canonical", "canonical_name"):
        if entry.get(key):
            return str(entry[key]).strip()
    return ""


def load_parties(path: Path) -> dict[str, list[str]]:
    """Load the party table, ignoring `_`-prefixed documentation keys.

    The table is a JSON file a human maintains, so it carries `_note`,
    `_election` and `_source_urls` alongside the data. Treating those as party
    stems makes every gloss look contradictory, which is what happened on the
    first run of this checker.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            out[key] = [value]
        elif isinstance(value, list):
            names = [str(v) for v in value if str(v).strip()]
            if names:
                out[key] = names
    return out


def find_contradictions(entries: list[dict], parties: dict[str, list[str]]) -> list[dict]:
    """Labels whose native stem names party A but whose gloss names party B."""
    # Build: folded stem -> set of folded correct glosses, and reverse.
    stem_to_gloss: dict[str, set[str]] = {}
    gloss_to_stem: dict[str, str] = {}
    for native, glosses in parties.items():
        ns = fold(native)
        stem_to_gloss[ns] = {fold(g) for g in glosses}
        for g in glosses:
            gloss_to_stem[fold(g)] = ns

    out: list[dict] = []
    for entry in entries:
        label = label_of(entry)
        m = re.match(r"^(?P<native>.+?)\s*\((?P<gloss>[^)]+)\)\s*(?P<tail>.*)$", label)
        if not m:
            continue
        native, gloss = fold(m["native"]), fold(m["gloss"])
        if not native or not gloss:
            continue

        # Which party does the native part name?
        native_party = next((s for s in stem_to_gloss if s and s in native), None)
        if native_party is None:
            continue

        # Which party does the gloss name? Longest match wins, so a gloss that
        # contains a shorter party name inside a longer one resolves correctly.
        gloss_party = None
        for g in sorted(gloss_to_stem, key=len, reverse=True):
            if g and g in gloss:
                gloss_party = gloss_to_stem[g]
                break
        if gloss_party is None:
            continue

        # Not a contradiction when the gloss is a correct alternative name for
        # the native party.
        if gloss_party == native_party:
            continue
        # Not a contradiction when the gloss merely *contains* the native
        # party's own correct name (e.g. "Sweden's Social Democratic Party").
        if any(correct in gloss for correct in stem_to_gloss.get(native_party, set())):
            continue

        native_key = next((k for k in parties if fold(k) == native_party), native_party)
        gloss_key = next((k for k in parties if fold(k) == gloss_party), gloss_party)
        out.append(
            {
                "label": label,
                "native_part_names": native_key,
                "correct_glosses_for_native": parties.get(native_key, []),
                "gloss_actually_names": gloss_key,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codebook", required=True)
    p.add_argument("--parties", required=True, help="JSON: native stem -> correct English name(s)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    book_path, parties_path = Path(args.codebook), Path(args.parties)
    for path in (book_path, parties_path):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 2
    try:
        entries = json.loads(book_path.read_text(encoding="utf-8")).get("entries", [])
        parties = load_parties(parties_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: bad JSON: {exc}", file=sys.stderr)
        return 2

    found = find_contradictions(entries, parties)
    if args.json:
        print(json.dumps(found, ensure_ascii=False, indent=2))
    elif found:
        print(f"{len(found)} contradictory party gloss(es) in {book_path.name}:")
        for row in found:
            print(f"  {row['label']!r}")
            print(f"      native part names : {row['native_part_names']}")
            print(f"      correct glosses   : {row['correct_glosses_for_native']}")
            print(f"      gloss actually is : {row['gloss_actually_names']}")
        print()
        print("Ambiguity is reported, never resolved. No codebook was modified.")
    else:
        print(f"no contradictory party glosses in {book_path.name}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
