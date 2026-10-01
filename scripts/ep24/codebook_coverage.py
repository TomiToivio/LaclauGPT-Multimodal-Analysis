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


#: Trailing tokens that name an entity *kind*, not an entity. A label like
#: "Centre Party" or "Brothers of Italy party" ends in one of these, so using it
#: as a surname key collapses every party into one group.
KIND_WORDS = frozenset(
    {
        "party", "parties", "coalition", "movement", "alliance", "list", "front",
        "group", "association", "organisation", "organization", "union", "bloc",
        "forum", "network", "institute", "institution", "foundation", "committee",
        "council", "agency", "ministry", "government", "parliament", "federation",
        "confederation", "league", "platform", "initiative", "campaign", "politics",
        "policy", "policies", "theme", "topic",
    }
)


def _fold(value: Any) -> str:
    """Casefold + strip accents + drop punctuation, for grouping only."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return {t for t in _fold(value).split() if len(t) > 2}


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    """The trailing *name* token, skipping trailing kind words.

    Taking the literal last token groups `Centre Party`, `Finns Party` and
    `Brothers of Italy party` under the key `party` — a false merge that hides
    the real fragmentation. Type words are skipped so the key is an actual name.
    """
    toks = [t for t in _fold(label).split() if len(t) > 3]
    toks = [t for t in toks if t not in KIND_WORDS]
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


def _iso2_of(path: Path) -> str | None:
    """The ISO2 code recorded *inside* a codebook file, if it has one.

    The books are not named consistently: `ep24_hr_private.json` and
    `ep24_es_private.json` use the ISO2 code, but `ep24_finland_private.json`
    and `ep24_poland_private.json` use the country name. Resolving by filename
    therefore fails for those two countries. The files record `country_code`
    internally, so read it rather than inferring it from the name.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("country_code")
    return str(value).strip().upper() if value else None


def _candidate_layers(root: Path, country: str) -> list[Path]:
    """Every per-country codebook under ``root`` matching ``country``.

    ``country`` may be an ISO2 code (`HR`) or a name (`Finland`). Matching is by
    the file's own `country_code` and `country` fields first, then by filename,
    so a book named after the country is still found from its ISO2 code.
    """
    wanted = country.strip().casefold()
    wanted_iso = country.strip().upper()

    matches: list[Path] = []
    for path in sorted(root.glob("ep24_*_private.json")):
        if path.name == "ep24_common_private.json":
            # The shared book is not a country layer; it is loaded as a base and
            # has no country_code of its own.
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = str(payload.get("country_code") or "").strip().upper()
        name = str(payload.get("country") or "").strip().casefold()
        if wanted_iso == code or wanted == name or wanted == path.stem.removeprefix("ep24_").removesuffix("_private"):
            matches.append(path)

    country_dir = root / "countries"
    if country_dir.is_dir():
        # `countries/<iso2>.json` is a second layer for some countries. Accept it
        # when either its name or its own country_code matches.
        for path in sorted(country_dir.glob("*.json")):
            if path.stem.casefold() == wanted or (
                path.stem.upper() == wanted_iso and _iso2_of(path) == wanted_iso
            ):
                matches.append(path)

    # Preserve order, drop duplicates.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in matches:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def audit(root: Path, country: str) -> dict[str, Any]:
    candidates = _candidate_layers(root, country)
    layers: dict[str, Any] = {}
    for path in candidates:
        entries = _load_entries(path)
        layers[path.name] = {
            "entries": entries,
            "alias_coverage": alias_coverage(entries),
            "kind_breakdown": kind_breakdown(entries),
            "entity_fragmentation": entity_fragmentation(entries),
            "theme_near_duplicates": theme_near_duplicates(entries),
        }
    if not layers:
        available = sorted(
            p.name for p in root.glob("ep24_*_private.json") if p.name != "ep24_common_private.json"
        )
        raise FileNotFoundError(
            f"no codebook found for {country!r} under {root}. "
            "Matching is by the file's own country_code/country field, then by filename. "
            "Books present: " + (", ".join(available) or "(none)")
        )
    return {"country_code": country.upper(), "root": str(root), "layers": layers}


def _print_human(report: dict[str, Any], *, top: int) -> None:
    print(f"EP24 codebook coverage report — {report['country_code']}")
    print(f"root: {report['root']}")
    for name, layer in report["layers"].items():
        cov = layer["alias_coverage"]
        print()
        print(f"=== {name} ===")
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
