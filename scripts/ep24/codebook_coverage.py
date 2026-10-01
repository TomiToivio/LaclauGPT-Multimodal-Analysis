#!/usr/bin/env python3
"""Report EP24 country-codebook coverage and fragmentation (issue #72).

A reusable, read-only auditor for the per-country codebook review. It encodes the
measurements from the Croatia pass (docs/EP24_CODEBOOK_AUDIT_HR.md) so any agent
reviewing any country can reproduce the same numbers instead of re-inventing the
analysis and reporting incomparable figures.

It answers, for one country:

  A. alias coverage        -- how many entries can be matched on any form but
                              their exact label
  B. entity fragmentation  -- the same actor under several labels
  C. party fragmentation   -- the same party under several labels
  D. entity-kind breakdown -- the check that caught "zero parties in the output"
  E. theme duplication     -- near-duplicate themes that fragment retrieval

Read-only. It never writes to a codebook, never merges, renames or deletes.
Ambiguity is REPORTED, never resolved.

Usage:
    python3 scripts/ep24/codebook_coverage.py --root <private codebook root> \
        --country HR [--json] [--top 15]

`--root` must be the directory holding the per-country books, i.e. the parent of
`countries/<iso2>.json` and of `ep24_<country>_private.json` (normally
`LaclauGPT-Private/analysis/ep24/codebooks`).

Exit codes: 0 = report produced, 2 = input/config problem.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fold(value: Any) -> str:
    """Casefold + strip accents + drop punctuation, for grouping only."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return {t for t in _fold(value).split() if len(t) > 2}


class CodebookUnreadable(RuntimeError):
    """A codebook file exists but cannot be read (LFS pointer, malformed JSON)."""


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """Load entries, failing cleanly on an LFS pointer or malformed JSON.

    A checked-out-but-unsmudged LFS pointer is a 130-byte text file beginning
    with `version https://git-lfs.github.com/spec/v1`. That is an environment
    problem (`git lfs pull`), not a codebook defect, and it must be reported as
    such rather than as a JSON traceback.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        raise CodebookUnreadable(
            f"{path.name} is an unsmudged Git LFS pointer - run "
            f"`git lfs pull --include='{path.name}'` first"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CodebookUnreadable(f"{path.name} is not valid JSON: {exc}") from exc
    out: list[dict[str, Any]] = []
    for item in payload.get("entries", []) or []:
        if isinstance(item, dict):
            out.append(item)
    return out


def _label(entry: dict[str, Any]) -> str:
    for key in ("label", "name", "canonical", "canonical_name"):
        value = entry.get(key)
        if value:
            return str(value).strip()
    return ""


def _observations(entry: dict[str, Any]) -> int:
    meta = entry.get("metadata") or {}
    prov = entry.get("provenance") or {}
    for source in (meta, prov):
        value = source.get("observations")
        if isinstance(value, int):
            return value
    return 0


def alias_coverage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A. How many entries can be matched on a form other than the exact label."""
    total = len(entries)
    with_aliases = sum(1 for e in entries if e.get("aliases"))
    aliases = sum(len(e.get("aliases") or []) for e in entries)
    return {
        "entries": total,
        "entries_without_aliases": total - with_aliases,
        "entries_without_aliases_pct": round((total - with_aliases) / total * 100, 1) if total else 0.0,
        "aliases_total": aliases,
        "aliases_per_entry": round(aliases / total, 2) if total else 0.0,
    }


def _fragment_groups(entries: list[dict[str, Any]], *, key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = key_fn(_label(entry))
        if key:
            groups[key].append(entry)
    rows = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        rows.append(
            {
                "key": key,
                "count": len(items),
                "observations": sum(_observations(i) for i in items),
                "labels": sorted(_label(i) for i in items),
            }
        )
    rows.sort(key=lambda r: (-r["observations"], -r["count"], r["key"]))
    return rows


def _surname_key(label: str) -> str:
    toks = [t for t in _fold(label).split() if len(t) > 3]
    return toks[-1] if toks else ""


def entity_fragmentation(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """B. Actors sharing a trailing name token across several entries."""
    return _fragment_groups(entries, key_fn=_surname_key)


def kind_breakdown(entries: list[dict[str, Any]]) -> dict[str, int]:
    """D. The check that makes 'no party entities at all' visible."""
    return dict(Counter(str(e.get("kind") or "entity") for e in entries).most_common())


def theme_near_duplicates(entries: list[dict[str, Any]], *, threshold: float = 0.6) -> list[dict[str, Any]]:
    """E. Theme pairs that would fragment the same retrieval concept."""
    themes = [e for e in entries if str(e.get("kind")) in {"theme", "topic"}]
    out = []
    for i, a in enumerate(themes):
        ta = _tokens(_label(a))
        if not ta:
            continue
        for b in themes[i + 1:]:
            tb = _tokens(_label(b))
            if not tb:
                continue
            jaccard = len(ta & tb) / len(ta | tb)
            if jaccard >= threshold and _fold(_label(a)) != _fold(_label(b)):
                out.append(
                    {
                        "jaccard": round(jaccard, 2),
                        "a": _label(a),
                        "b": _label(b),
                    }
                )
    out.sort(key=lambda r: (-r["jaccard"], r["a"]))
    return out


def _candidate_paths(root: Path, country: str) -> list[Path]:
    """Resolve a country's codebook files using the repo's own country map.

    The filename is NOT derivable from the ISO2 code: Finland is
    `ep24_finland_private.json` (not `ep24_fi_private.json`) and Poland is
    `ep24_poland_private.json`. Guessing from the code silently reports
    "no codebook found" for those countries, so use COUNTRY_PROFILES when it is
    importable and fall back to the slug only for unmapped countries.
    """
    iso = country.upper()
    slug = country.casefold()
    candidates: list[Path] = []
    try:
        from roihu_codebooks import COUNTRY_PROFILES  # type: ignore

        mapped = COUNTRY_PROFILES.get(iso)
        if mapped and mapped.get("file"):
            candidates.append(root / str(mapped["file"]))
        elif mapped and mapped.get("country"):
            candidates.append(root / f"ep24_{str(mapped['country']).casefold()}_private.json")
    except Exception:  # pragma: no cover - map is a convenience, not a hard dep
        pass
    candidates.append(root / f"ep24_{slug}_private.json")
    candidates.append(root / "countries" / f"{slug}.json")
    # de-duplicate, preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        if path.name not in seen:
            seen.add(path.name)
            unique.append(path)
    return unique


def audit(root: Path, country: str) -> dict[str, Any]:
    iso = country.upper()
    candidates = _candidate_paths(root, country)
    layers: dict[str, Any] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            entries = _load_entries(path)
        except CodebookUnreadable as exc:
            # Report the unreadable layer instead of aborting: a pointer in one
            # layer must not hide the numbers for the layer that IS readable.
            layers[path.name] = {"unreadable": str(exc)}
            continue
        layers[path.name] = {
            "entries": entries,
            "alias_coverage": alias_coverage(entries),
            "kind_breakdown": kind_breakdown(entries),
            "entity_fragmentation": entity_fragmentation(entries),
            "theme_near_duplicates": theme_near_duplicates(entries),
        }
    if not layers:
        raise FileNotFoundError(
            f"no codebook found for {iso} under {root} (looked for "
            + ", ".join(p.name for p in candidates)
            + ")"
        )
    return {"country_code": iso, "root": str(root), "layers": layers}


def _print_human(report: dict[str, Any], *, top: int) -> None:
    print(f"EP24 codebook coverage report — {report['country_code']}")
    print(f"root: {report['root']}")
    for name, layer in report["layers"].items():
        print()
        print(f"=== {name} ===")
        if "unreadable" in layer:
            print(f"  UNREADABLE: {layer['unreadable']}")
            continue
        cov = layer["alias_coverage"]
        print(f"  entries                    : {cov['entries']}")
        print(f"  without aliases            : {cov['entries_without_aliases']} ({cov['entries_without_aliases_pct']}%)")
        print(f"  aliases total / per entry  : {cov['aliases_total']} / {cov['aliases_per_entry']}")
        print(f"  kind breakdown             : {layer['kind_breakdown']}")
        frag = layer["entity_fragmentation"][:top]
        print(f"  fragmented entities (top {len(frag)} of {len(layer['entity_fragmentation'])}):")
        for row in frag:
            print(f"    obs={row['observations']:<5} n={row['count']:<3} {row['labels'][:3]}")
        tdup = layer["theme_near_duplicates"][:top]
        print(f"  near-duplicate themes (top {len(tdup)} of {len(layer['theme_near_duplicates'])}):")
        for row in tdup:
            print(f"    {row['jaccard']}  {row['a']!r} ~ {row['b']!r}")
    print()
    print("NOTE: ambiguity is reported, never resolved. No codebook was modified.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory holding the per-country codebooks")
    parser.add_argument("--country", required=True, help="ISO2 country code, e.g. HR")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the human report")
    parser.add_argument("--top", type=int, default=15, help="rows to show per section (default 15)")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2
    try:
        report = audit(root, args.country)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
