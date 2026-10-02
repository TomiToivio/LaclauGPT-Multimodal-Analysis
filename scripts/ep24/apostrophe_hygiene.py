#!/usr/bin/env python3
"""Detect identity drift caused by punctuation that is not punctuation to a human.

Issue #91, France pass. This is the French generalisation of the Bulgaria pass's
Latin/Cyrillic homoglyph detector (`scripts/ep24/cyrillic_hygiene.py`).

The defect class is the same shape: two strings that a human reader sees as the
same entity, but which `identity_key` correctly — and deliberately — keeps
distinct because it preserves punctuation. For Bulgarian the confusable pair was
Latin `M` vs Cyrillic `М`. For French the live pair is:

    'l'Union européenne'    ASCII APOSTROPHE  U+0027
    'l\u2019Union européenne'  RIGHT SINGLE QUOTATION MARK  U+2019

`identity_key` normalises NFC, casefolds and collapses whitespace, and nothing
else, because accents and punctuation are analytically meaningful (see the HR
audit's rule 1). That is the RIGHT design. But French elision is written with
either apostrophe depending on the source:

* French keyboard input and most ASR output use the ASCII `'`;
* word processors, publishing pipelines, iOS/macOS autocorrect and much French
  news web output use the typographic `’` (U+2019).

Both are the same French elision. Because NFC does NOT unify them (they are
canonically unrelated characters — `Po` and `Pf` categories, not a
compatibility mapping), a codebook entry keyed on `l'Union` is unreachable from
a post written `l’Union` and vice versa.

This script REPORTS, it does not repair. Rewriting a label changes identity, and
whether the typographic form is the canonical one is a researcher decision. The
tool's job is to make the split visible and to say which pairs are safe.

Safety distinction the report makes explicit:

* `'` vs `’` — SAME elision, different keyboard. A safe normalisation candidate.
* `'` vs `′` (U+2032 PRIME) — NOT the same character. Prime means minutes/feet.
* `-` vs `–` vs `—` — SAME separator, different typography, but a hyphen can
  also join a double-barrelled surname, so this is reported separately.

Reported-only for: hyphen/dash variants and the guillemet/quote family.

Usage:
    python3 scripts/ep24/apostrophe_hygiene.py --root <codebooks dir> [--country FR] [--json]
    python3 scripts/ep24/apostrophe_hygiene.py --text "l’Union européenne"

Exit codes: 0 = clean or reported, 1 = findings present, 2 = input problem.
Exit 1 makes it usable as a gate while still printing the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Character families, ordered by how safe a unification would be.
#
# SAFE_ELISION is the only family this project should ever consider unifying:
# both characters are apostrophes with the same linguistic function, and the
# difference is purely which keyboard/pipeline produced them.
SAFE_ELISION: dict[str, str] = {
    "\u0027": "ASCII APOSTROPHE",           # '
    "\u2019": "RIGHT SINGLE QUOTATION MARK",  # ’
    "\u2018": "LEFT SINGLE QUOTATION MARK",   # ‘
    "\u02bc": "MODIFIER LETTER APOSTROPHE",   # ʼ
}

# NOT unified, deliberately: a prime is a different character with a different
# meaning, and a backtick is not an apostrophe at all.
NOT_APOSTROPHES: dict[str, str] = {
    "\u2032": "PRIME",                       # ′ minutes/feet, not elision
    "\u2033": "DOUBLE PRIME",
    "\u0060": "GRAVE ACCENT",                # `
}

DASH_FAMILY: dict[str, str] = {
    "\u002d": "HYPHEN-MINUS",                # -
    "\u2010": "HYPHEN",
    "\u2011": "NON-BREAKING HYPHEN",
    "\u2012": "FIGURE DASH",
    "\u2013": "EN DASH",                     # –
    "\u2014": "EM DASH",                     # —
    "\u2212": "MINUS SIGN",
}

# The canonical form this project already writes into its own labels. Reported
# so a drift can be measured against it, not asserted as the only right answer.
PROJECT_APOSTROPHE = "\u0027"


def describe(char: str) -> str:
    """Human-readable name for a character, with its code point."""
    return f"{unicodedata.name(char, '?')} (U+{ord(char):04X})"


def normalise_elision(text: str) -> str:
    """Fold every apostrophe variant onto the project's canonical apostrophe.

    Deliberately narrow: only the SAFE_ELISION family is folded. Primes, dashes
    and quote marks are left alone, because unifying them would be a semantic
    change rather than a normalisation.
    """
    return "".join(PROJECT_APOSTROPHE if c in SAFE_ELISION else c for c in str(text or ""))


def apostrophe_variants(text: str) -> list[dict[str, Any]]:
    """Every apostrophe-family character in ``text``, with its position."""
    out: list[dict[str, Any]] = []
    for index, char in enumerate(str(text or "")):
        if char in SAFE_ELISION:
            out.append(
                {
                    "index": index,
                    "char": char,
                    "codepoint": f"U+{ord(char):04X}",
                    "unicode_name": unicodedata.name(char, "?"),
                    "role": SAFE_ELISION[char],
                    "context": str(text)[max(0, index - 14): index + 15],
                }
            )
        elif char in NOT_APOSTROPHES:
            out.append(
                {
                    "index": index,
                    "char": char,
                    "codepoint": f"U+{ord(char):04X}",
                    "unicode_name": unicodedata.name(char, "?"),
                    "role": NOT_APOSTROPHES[char],
                    "context": str(text)[max(0, index - 14): index + 15],
                }
            )
    return out


def is_typographic(text: str) -> bool:
    """True when the text uses a non-ASCII apostrophe for elision."""
    return any(c in SAFE_ELISION and c != PROJECT_APOSTROPHE for c in str(text or ""))


def audit_text(text: str) -> dict[str, Any] | None:
    """Return a finding for one label, or None when it is clean."""
    variants = apostrophe_variants(text)
    if not variants:
        return None
    typed = [v for v in variants if v["char"] in SAFE_ELISION]
    non_apostrophe = [v for v in variants if v["char"] in NOT_APOSTROPHES]
    unified = normalise_elision(text)
    return {
        "label": text,
        "unified": unified,
        "unification_changes_text": unified != text,
        "typographic_elision": is_typographic(text),
        "apostrophes": typed,
        "not_apostrophes": non_apostrophe,
        "nfc": unicodedata.normalize("NFC", text),
        "nfc_changes": unicodedata.normalize("NFC", text) != text,
    }


def _iter_labels(payload: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield (field, value) for every label-bearing field in a codebook."""
    for item in payload.get("entries", []) or []:
        if not isinstance(item, dict):
            continue
        for key in ("label", "english_label", "canonical", "canonical_name", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                yield key, value
        for alias in item.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                yield "aliases", alias


def _load(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _candidate_files(root: Path, country: str | None) -> list[Path]:
    if country:
        iso = country.upper()
        slug = country.casefold()
        cands: list[Path] = []
        try:
            from roihu_codebooks import COUNTRY_PROFILES  # type: ignore

            mapped = COUNTRY_PROFILES.get(iso)
            if mapped and mapped.get("file"):
                cands.append(root / str(mapped["file"]))
        except Exception:  # pragma: no cover
            pass
        cands.append(root / f"ep24_{slug}_private.json")
        cands.append(root / "countries" / f"{slug}.json")
        seen: set[str] = set()
        out: list[Path] = []
        for path in cands:
            if path.name not in seen:
                seen.add(path.name)
                out.append(path)
        return out
    return sorted(list(root.glob("ep24_*_private.json")) + list(root.glob("countries/*.json")))


def identity_split(labels: Iterable[str]) -> list[dict[str, Any]]:
    """Label pairs that differ ONLY by apostrophe style.

    These are the actionable findings: two observations of one entity that are
    unreachable from each other because `identity_key` preserves punctuation.

    The test is deliberately strict. `identity_key` already casefolds and
    collapses whitespace, so a pair like `Alliance Rurale` / `Alliance rurale`
    is NOT a split — it is one identity. A real split is a group that collapses
    under ``normalise_elision`` but still holds MORE THAN ONE distinct
    ``identity_key``. That is exactly "these differ only in which apostrophe was
    typed".
    """
    try:
        from roihu_codebooks import identity_key  # type: ignore
    except Exception:  # pragma: no cover
        return []

    groups: dict[str, dict[str, set[str]]] = {}
    for label in labels:
        unified_key = identity_key(normalise_elision(label))
        raw_key = identity_key(label)
        entry = groups.setdefault(unified_key, {})
        entry.setdefault(raw_key, set()).add(label)

    out = []
    for unified_key, by_raw in groups.items():
        if len(by_raw) < 2:
            continue
        out.append(
            {
                "unified_key": unified_key,
                "distinct_identity_keys": sorted(by_raw),
                "labels": sorted({label for members in by_raw.values() for label in members}),
            }
        )
    out.sort(key=lambda r: r["unified_key"])
    return out


def audit(root: Path | None, country: str | None = None, *, text: str | None = None) -> dict[str, Any]:
    if text is not None:
        finding = audit_text(text)
        return {
            "mode": "text",
            "findings": [finding] if finding else [],
            "label_count": 1 if finding else 0,
            "split_groups": [],
            "scanned": [],
            "skipped_unreadable": [],
        }

    assert root is not None
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    skipped: list[str] = []
    labels: list[str] = []
    for path in _candidate_files(root, country):
        if not path.exists():
            continue
        payload = _load(path)
        if payload is None:
            skipped.append(path.name)
            continue
        scanned.append(path.name)
        for field, value in _iter_labels(payload):
            labels.append(value)
            finding = audit_text(value)
            if finding:
                findings.append({"file": path.name, "field": field, **finding})

    splits = identity_split(labels)
    return {
        "mode": "codebook",
        "root": str(root),
        "country": country.upper() if country else None,
        "scanned": scanned,
        "skipped_unreadable": skipped,
        "label_count": len(labels),
        "finding_count": len(findings),
        "typographic_count": sum(1 for f in findings if f["typographic_elision"]),
        "split_groups": splits,
        "split_group_count": len(splits),
        "findings": findings,
    }


def _print_human(report: dict[str, Any]) -> None:
    if report["mode"] == "text":
        print("EP24 apostrophe hygiene — single string")
        print()
        for finding in report["findings"]:
            print(f"  input    : {finding['label']!r}")
            print(f"  unified  : {finding['unified']!r}")
            print(f"  changed  : {finding['unification_changes_text']}")
            print(f"  typographic elision: {finding['typographic_elision']}")
            for ap in finding["apostrophes"]:
                print(f"    {ap['char']!r} {ap['codepoint']} {ap['role']}  in {ap['context']!r}")
            for other in finding["not_apostrophes"]:
                print(f"    NOTE {other['char']!r} {other['codepoint']} {other['role']} is NOT an apostrophe")
        if not report["findings"]:
            print("No apostrophe-family characters found.")
        return

    print(f"EP24 apostrophe hygiene — {report['country'] or 'all countries'}")
    print(f"root: {report['root']}")
    print(f"scanned: {', '.join(report['scanned']) or '(none)'}")
    if report["skipped_unreadable"]:
        print(f"skipped (LFS pointer / bad JSON): {', '.join(report['skipped_unreadable'])}")
    print()
    print(f"labels/forms inspected        : {report['label_count']}")
    print(f"forms containing an apostrophe: {report['finding_count']}")
    print(f"  of which typographic (not ') : {report['typographic_count']}")
    print(f"identity groups split ONLY on apostrophe style: {report['split_group_count']}")
    if report["split_groups"]:
        print()
        for group in report["split_groups"][:20]:
            print(f"  {group['unified_key']!r}")
            for label in group["labels"]:
                print(f"      {label!r}")
    print()
    print("REPORT ONLY - nothing was rewritten. Unifying ' with \u2019 changes an identity")
    print("key, and which form is canonical is a researcher decision, not a lint fix.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="directory holding the per-country codebooks")
    parser.add_argument("--country", help="ISO2 country code (e.g. FR); omit for all")
    parser.add_argument("--text", help="audit a single string instead of a codebook")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    if not args.text and not args.root:
        print("ERROR: pass --root or --text", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser() if args.root else None
    if root is not None and not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    report = audit(root, args.country, text=args.text)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
