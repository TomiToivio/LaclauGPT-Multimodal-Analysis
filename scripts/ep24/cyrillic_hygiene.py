#!/usr/bin/env python3
"""Detect Latin/Cyrillic mixed-script (homoglyph) defects in EP24 codebooks.

Issue #91, Bulgaria pass. Bulgaria is the only Cyrillic-script country in the
EP24 set, and mixing writing systems inside one label produces characters that
RENDER correctly and therefore survive human review while defeating exact
matching forever.

The observed live example:

    'European Parliament (SEМ)'      <- U+041C CYRILLIC CAPITAL LETTER EM
    identity_key(...) = 'european parliament (seм)'   (Cyrillic м)
    a correctly typed 'SEM' has   = 'european parliament (sem)'   (Latin m)
    -> never equal, and NFC does not fix it (the characters are canonically
       unrelated, so normalisation cannot know which was meant)

This script REPORTS, it does not repair. Rewriting a researcher-grounded label is
a review decision, not a lint fix: the tool cannot know whether the author meant
Latin `SEP` or Cyrillic `СЕП`.

Usage:
    python3 scripts/ep24/cyrillic_hygiene.py --root <codebooks dir> [--country BG] [--json]

Exit codes: 0 = clean or reported, 1 = mixed-script findings present, 2 = input
problem. Exit 1 makes it usable as a quality gate while still printing the report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
LATIN = re.compile(r"[A-Za-z]")

# Latin letters whose Cyrillic siblings are visually identical. Only the pairs
# that actually cause silent mismatches are listed; this is a detector, not a
# full Unicode confusables table.
CONFUSABLES = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
}


def script_of(char: str) -> str:
    """Return 'cyrillic', 'latin', or 'other' for one character."""
    if CYRILLIC.match(char):
        return "cyrillic"
    if LATIN.match(char):
        return "latin"
    return "other"


def find_homoglyphs(text: str) -> list[dict[str, Any]]:
    """Cyrillic lookalikes of Latin letters, reported only where they cause a defect.

    Restricted to tokens that mix scripts. Inside a wholly Cyrillic token a
    character like `о` is simply correct Cyrillic, not a homoglyph, and reporting
    it would bury the real findings in noise.
    """
    defective = {
        token
        for token in re.split(r"[^\w]+", text, flags=re.UNICODE)
        if token and CYRILLIC.search(token) and LATIN.search(token)
    }
    out: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char not in CONFUSABLES:
            continue
        token = next(
            (t for t in defective if text[max(0, index - len(t)): index + len(t) + 1].find(t) >= 0),
            "",
        )
        if not token:
            continue
        out.append(
            {
                "index": index,
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "unicode_name": unicodedata.name(char, "?"),
                "latin_lookalike": CONFUSABLES[char],
                "token": token,
                "context": text[max(0, index - 12): index + 13],
            }
        )
    return out


def is_mixed_script(text: str) -> bool:
    """True when a SINGLE TOKEN mixes scripts — the actual defect.

    A label may legitimately pair the two scripts, which is the normal bilingual
    form in this codebook:

        'DPS (ДПС)'                          Latin token + Cyrillic token -> FINE
        'Солидарна България (Solidarity ...)' -> FINE

    A defect is a script boundary INSIDE one token, which happens when a character
    is typed in the wrong writing system:

        'European Parliament (SEМ)'   -> 'SEМ' is S,E + Cyrillic М -> DEFECT

    Reporting the bilingual form as a defect would be noise, and a noisy gate gets
    ignored. So: split on anything that is not a letter, and inspect each token.
    """
    for token in re.split(r"[^\w]+", text, flags=re.UNICODE):
        if token and CYRILLIC.search(token) and LATIN.search(token):
            return True
    return False


def audit_text(text: str) -> dict[str, Any] | None:
    """Return a finding for one label, or None when it is clean."""
    if not is_mixed_script(text):
        return None
    return {
        "label": text,
        "mixed_script": True,
        "homoglyphs": find_homoglyphs(text),
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


def audit(root: Path, country: str | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    skipped: list[str] = []
    for path in _candidate_files(root, country):
        if not path.exists():
            continue
        payload = _load(path)
        if payload is None:
            skipped.append(path.name)
            continue
        scanned.append(path.name)
        for field, value in _iter_labels(payload):
            finding = audit_text(value)
            if finding:
                findings.append({"file": path.name, "field": field, **finding})
    return {
        "root": str(root),
        "country": country.upper() if country else None,
        "scanned": scanned,
        "skipped_unreadable": skipped,
        "mixed_script_count": len(findings),
        "findings": findings,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"EP24 Latin/Cyrillic script hygiene — {report['country'] or 'all countries'}")
    print(f"root: {report['root']}")
    print(f"scanned: {', '.join(report['scanned']) or '(none)'}")
    if report["skipped_unreadable"]:
        print(f"skipped (LFS pointer / bad JSON): {', '.join(report['skipped_unreadable'])}")
    print()
    if not report["findings"]:
        print("No mixed-script labels found.")
        return
    print(f"FOUND {report['mixed_script_count']} mixed-script label(s):")
    for finding in report["findings"]:
        print()
        print(f"  file : {finding['file']}")
        print(f"  field: {finding['field']}")
        print(f"  label: {finding['label']!r}")
        if finding["nfc_changes"]:
            print("         (NFC would change this - re-normalise the file)")
        for hom in finding["homoglyphs"]:
            print(
                f"    {hom['char']!r} {hom['codepoint']} {hom['unicode_name']}"
                f"  looks like Latin {hom['latin_lookalike']!r}"
                f"  in {hom['context']!r}"
            )
    print()
    print("REPORT ONLY - nothing was rewritten. Replacing a Cyrillic lookalike with")
    print("its Latin counterpart changes a researcher-grounded label and needs review:")
    print("the tool cannot know which writing system the author intended.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory holding the per-country codebooks")
    parser.add_argument("--country", help="ISO2 country code (e.g. BG); omit for all")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    report = audit(root, args.country)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 1 if report["mixed_script_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
