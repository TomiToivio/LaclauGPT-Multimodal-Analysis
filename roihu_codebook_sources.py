"""Load the private ``codebook_sources`` replacement workbooks into registry entries.

The workbooks are researcher-maintained ``old`` -> ``new`` replacement tables, not
JSON registries:

    file            sheet    shape         meaning
    --------------  -------  ------------  --------------------------------------
    entities.xlsx   Sheet3   old, new      surface mention -> canonical entity
    persons.xlsx    Sheet3   old, new      surface mention -> canonical person
    themes.xlsx     Taul1    old, new      surface mention -> canonical theme

Two structural properties drive the parsing, and both were measured against the
real files rather than assumed:

1. ``new`` usually embeds the disambiguation in parentheses --
   ``Canonical Name (Party; Country)``. Taking ``new`` verbatim as the canonical
   label means a plain mention of that person does not match their own canonical
   form, so the parenthetical is split into party and country and stored
   separately as structured fields.
2. The grouping *is* the alias set: every ``old`` sharing a ``new`` is an alias of
   that canonical entity. Themes collapse thousands of surface forms into a small
   canonical set (one theme carries ~160 alias forms), so this is real alias data.

This module is public and data-agnostic. Real workbook contents stay in
LaclauGPT-Private; tests use small invented fixtures with the same structure.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from roihu_codebooks import CodebookEntry, _entry_id, identity_key

#: Workbook file name -> (kind, entity_type). The kind vocabulary matches
#: ``roihu_codebooks``/``roihu_identity`` so the merged registry stays consistent
#: with the JSON profile layer.
WORKBOOK_KINDS: dict[str, tuple[str, str]] = {
    "entities.xlsx": ("entity", "ORGANIZATION"),
    "persons.xlsx": ("person", "PERSON"),
    "themes.xlsx": ("topic", "THEME"),
}

#: Alternative file names accepted for the same content, so a renamed artifact
#: does not silently stop being loaded.
WORKBOOK_ALIASES: dict[str, str] = {
    "themed.xlsx": "themes.xlsx",
    "entities_.xlsx": "entities.xlsx",
    "persons_.xlsx": "persons.xlsx",
}

#: Which entity type is the more specific statement about an actor. Used when the
#: same canonical name appears in more than one workbook (measured: every row of
#: persons.xlsx also appears in entities.xlsx, so the filename-derived
#: ORGANIZATION type would otherwise mislabel all 418 people).
_TYPE_PRECEDENCE: dict[str, int] = {
    "PERSON": 3,
    "ORGANIZATION": 2,
    "THEME": 1,
}

_PAREN_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<inner>[^)]*)\)\s*$")


def _frame_for(path: Path, sheet: str | None) -> pd.DataFrame:
    """Read a workbook or CSV, tolerating a headerless two-column sheet."""
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    elif path.suffix.casefold() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, dtype="string", keep_default_na=False, sheet_name=sheet or 0)
    else:
        raise ValueError(f"unsupported codebook_source format: {path.suffix}")
    lowered = {str(c).strip().casefold() for c in frame.columns}
    if {"old", "new"} <= lowered:
        return frame
    # No ``old``/``new`` header: re-read treating the first row as data, because a
    # headerless replacement sheet is still a valid source and silently returning
    # the wrong rows would drop one pair from every such file.
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False, header=None)
    return pd.read_excel(path, dtype="string", keep_default_na=False, sheet_name=sheet or 0, header=None)

#: Country names as they appear in the parenthetical, mapped to EP24 codes.
COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "finland": "FI", "sweden": "SE", "poland": "PL", "portugal": "PT",
    "germany": "DE", "spain": "ES", "hungary": "HU", "croatia": "HR",
    "france": "FR", "bulgaria": "BG",
    "suomi": "FI", "sverige": "SE", "polska": "PL", "deutschland": "DE",
    "españa": "ES", "espana": "ES", "magyarország": "HU", "hrvatska": "HR",
    "frankrike": "FR", "frankreich": "FR", "frança": "FR", "francia": "FR",
}


@dataclass
class ReplacementRow:
    """One ``old`` -> ``new`` row, parsed."""

    surface: str
    canonical_raw: str
    canonical_name: str = ""
    party: str = ""
    country: str = ""
    extra: list[str] = field(default_factory=list)


@dataclass
class RegistryGroup:
    """One canonical entity with its alias surface forms."""

    canonical_raw: str
    canonical_name: str
    kind: str
    entity_type: str
    party: str = ""
    country: str = ""
    aliases: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    def to_entry(self, *, source: str, layer: str = "researcher") -> CodebookEntry:
        """Convert to the shared ``CodebookEntry`` so both layers merge cleanly.

        ``origin`` is ``researcher_private``: these are researcher-maintained
        canonical mappings, which ``_normalize_item`` treats as human-locked.
        """
        label = self.canonical_name or self.canonical_raw
        metadata: dict[str, Any] = {
            "source_workbook": source,
            "canonical_raw": self.canonical_raw,
        }
        if self.party:
            metadata["party"] = self.party
        if self.extra:
            metadata["canonical_extra"] = list(self.extra)
        return CodebookEntry(
            entry_id=_entry_id(
                {"canonical_id": f"XLSX:{self.kind}:{label}:{self.country}"},
                self.kind, label, self.country,
            ),
            kind=self.kind,
            label=label,
            aliases=[a for a in dict.fromkeys(self.aliases) if a],
            country=self.country,
            entity_type=self.entity_type,
            review_state="CANONICAL",
            origin="researcher_private",
            locked=True,
            layer=layer,
            metadata=metadata,
        )


def parse_canonical(raw: str) -> tuple[str, str, str, list[str]]:
    """Split ``Name (Party; Country)`` into name, party, country, extras.

    Returns ``(name, party, country, extra_parts)``. A value without parentheses is
    returned unchanged as the name. Country is recognised by name so the result is
    an EP24 country code; unrecognised trailing parts are preserved in ``extra``
    rather than dropped, because silently discarding researcher annotation is the
    failure mode this whole layer exists to prevent.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return "", "", "", []
    match = _PAREN_RE.match(text)
    if not match:
        return text, "", "", []
    name = " ".join(match.group("name").split())
    parts = [p.strip() for p in match.group("inner").split(";") if p.strip()]
    party = ""
    country = ""
    extra: list[str] = []
    for part in parts:
        code = COUNTRY_NAME_TO_CODE.get(part.casefold())
        if code and not country:
            country = code
            continue
        if not party and part:
            party = part
            continue
        extra.append(part)
    return name, party, country, extra


def read_replacement_rows(path: str | Path, *, sheet: str | None = None) -> list[ReplacementRow]:
    """Read a workbook (or CSV) with ``old``/``new`` columns into parsed rows.

    The header is matched case-insensitively and the first two columns are used as
    a fallback, because the researcher workbooks use different sheet names
    (``Sheet3``, ``Taul1``) and a headerless variant must not crash the load.
    """
    path = Path(path)
    frame = _frame_for(path, sheet)

    lowered = {str(c).strip().casefold(): c for c in frame.columns}
    old_col = lowered.get("old")
    new_col = lowered.get("new")
    if old_col is None or new_col is None:
        if len(frame.columns) < 2:
            raise ValueError(f"{path.name}: expected 'old'/'new' columns")
        old_col, new_col = list(frame.columns)[:2]

    rows: list[ReplacementRow] = []
    for old, new in zip(frame[old_col].tolist(), frame[new_col].tolist()):
        surface = " ".join(str(old or "").split())
        raw = " ".join(str(new or "").split())
        if not surface or not raw:
            continue
        name, party, country, extra = parse_canonical(raw)
        rows.append(ReplacementRow(
            surface=surface, canonical_raw=raw, canonical_name=name,
            party=party, country=country, extra=extra,
        ))
    return rows


def group_rows(rows: Iterable[ReplacementRow]) -> list[RegistryGroup]:
    """Collapse replacement rows into canonical groups with alias lists.

    Grouping key is the *canonical name* (identity-normalized) **together with the
    country**, rather than the raw ``new`` string. Two consequences, both required:

    * two rows that differ only in their parenthetical annotation join one canonical
      entity instead of becoming duplicates;
    * the same name in two different countries stays two entities. The issue
      requires that explicitly ("same/similar names in two countries that remain
      separate"), and keying on the name alone silently merged them -- measured on
      a fixture with one surname in HR and DE, which collapsed to a single HR entry
      so the DE mention could not resolve at all.

    Party and country are merged across the group; conflicting values are kept via
    ``extra`` rather than dropped.
    """
    groups: dict[tuple[str, str], RegistryGroup] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        name = row.canonical_name or parse_canonical(row.canonical_raw)[0] or row.canonical_raw
        country = row.country or parse_canonical(row.canonical_raw)[2]
        key = (identity_key(name), country)
        if not key[0]:
            continue
        if key not in groups:
            groups[key] = RegistryGroup(
                canonical_raw=row.canonical_raw,
                canonical_name=name,
                kind="", entity_type="",
                party=row.party, country=country,
            )
            order.append(key)
        group = groups[key]
        # The surface form is an alias of the canonical entity. The canonical name
        # itself is also recorded as an alias when it differs from the surface, so a
        # lookup by the bare canonical name hits without a special case.
        for form in (row.surface, name):
            if form and form not in group.aliases:
                group.aliases.append(form)
        if not group.party and row.party:
            group.party = row.party
        elif row.party and row.party != group.party and row.party not in group.extra:
            group.extra.append(row.party)
        if not group.country and country:
            group.country = country
        group.extra.extend(x for x in row.extra if x not in group.extra)
    return [groups[k] for k in order]


def load_workbook_registry(
    path: str | Path,
    *,
    source_name: str | None = None,
    layer: str = "researcher",
) -> tuple[list[CodebookEntry], dict[str, Any]]:
    """Load one replacement workbook into registry entries plus a load report."""
    path = Path(path)
    canonical_source = WORKBOOK_ALIASES.get(path.name.casefold(), path.name.casefold())
    if canonical_source not in WORKBOOK_KINDS:
        raise ValueError(
            f"{path.name}: unknown codebook_source workbook; expected one of "
            + ", ".join(sorted(WORKBOOK_KINDS))
        )
    kind, entity_type = WORKBOOK_KINDS[canonical_source]
    rows = read_replacement_rows(path)
    groups = group_rows(rows)
    for group in groups:
        group.kind = kind
        group.entity_type = entity_type
    entries = [
        g.to_entry(source=source_name or canonical_source, layer=layer) for g in groups
    ]
    report = {
        "workbook": canonical_source,
        "kind": kind,
        "rows": len(rows),
        "canonical_entities": len(groups),
        "alias_forms": sum(len(g.aliases) for g in groups),
        "with_country": sum(1 for g in groups if g.country),
        "countries": sorted({g.country for g in groups if g.country}),
    }
    return entries, report


def find_source_workbooks(root: str | Path) -> dict[str, Path]:
    """Locate the known workbooks under a codebook_sources directory."""
    base = Path(root)
    found: dict[str, Path] = {}
    if not base.exists():
        return found
    for candidate in sorted(base.iterdir()):
        if not candidate.is_file():
            continue
        name = candidate.name.casefold()
        canonical = WORKBOOK_ALIASES.get(name, name)
        if canonical in WORKBOOK_KINDS:
            found.setdefault(canonical, candidate)
    return found


def load_registry_from_dir(root: str | Path, *, layer: str = "researcher") -> tuple[list[CodebookEntry], dict[str, Any]]:
    """Load every known workbook found under ``root`` into one entry list.

    The workbooks overlap: measured against the real files, ``persons.xlsx`` is a
    strict *subset* of ``entities.xlsx`` (all 418 person rows also appear in the
    entity rows). A naive per-file load therefore produces two entries per person
    -- one typed ORGANIZATION from the filename rule and one typed PERSON -- which
    would make every person unreachable under a person-scoped lookup and split
    their aliases across two ids.

    Groups are therefore merged by normalized canonical name, with type precedence
    ``PERSON`` > ``ORGANIZATION`` > ``THEME``: the more specific statement about a
    real-world actor wins, and every contributing workbook is recorded in
    ``metadata["source_workbook"]`` as a list so provenance is not lost.
    """
    entries: list[CodebookEntry] = []
    reports: list[dict[str, Any]] = []
    merged: dict[tuple[str, str], CodebookEntry] = {}
    order: list[tuple[str, str]] = []

    for name, path in sorted(find_source_workbooks(root).items()):
        try:
            work_entries, report = load_workbook_registry(path, layer=layer)
        except Exception as exc:  # a broken workbook must not abort the registry
            reports.append({"workbook": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reports.append(report)
        for entry in work_entries:
            key = (identity_key(entry.label), entry.country)
            existing = merged.get(key)
            if existing is None:
                merged[key] = entry
                order.append(key)
                continue
            # Prefer the more specific type; keep the union of aliases.
            if _TYPE_PRECEDENCE.get(entry.entity_type, 0) > _TYPE_PRECEDENCE.get(existing.entity_type, 0):
                existing.entity_type = entry.entity_type
                existing.kind = entry.kind
            for alias in entry.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            sources = existing.metadata.setdefault("source_workbook", [])
            if isinstance(sources, str):
                sources = [sources]
                existing.metadata["source_workbook"] = sources
            for src in ([entry.metadata.get("source_workbook")] if isinstance(entry.metadata.get("source_workbook"), str)
                        else (entry.metadata.get("source_workbook") or [])):
                if src and src not in sources:
                    sources.append(src)
            if entry.metadata.get("party") and not existing.metadata.get("party"):
                existing.metadata["party"] = entry.metadata["party"]

    entries = [merged[k] for k in order]
    totals = {
        "canonical_entities": len(entries),
        "alias_forms": sum(len(e.aliases) for e in entries),
        "by_type": {},
    }
    for entry in entries:
        totals["by_type"][entry.entity_type] = totals["by_type"].get(entry.entity_type, 0) + 1
    return entries, {"workbooks": reports, "totals": totals}
