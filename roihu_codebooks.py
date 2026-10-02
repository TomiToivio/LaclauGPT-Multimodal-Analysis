#!/usr/bin/env python3
"""Load and retrieve private bilingual EP24 codebooks on CSC Roihu.

Operational codebooks stay outside this public repository. This loader supports
existing EP24 private JSON shapes and adds deterministic, auditable context
selection without treating background knowledge as evidence from a post/video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
COUNTRY_PROFILES = {
    "FI": {"country": "Finland", "languages": ["fi", "sv", "en"], "file": "ep24_finland_private.json"},
    "SE": {"country": "Sweden", "languages": ["sv", "en"], "file": "ep24_se_private.json"},
    "PL": {"country": "Poland", "languages": ["pl", "en"], "file": "ep24_poland_private.json"},
    "PT": {"country": "Portugal", "languages": ["pt", "en"], "file": "ep24_pt_private.json"},
    "DE": {"country": "Germany", "languages": ["de", "en"], "file": "ep24_de_private.json"},
    "ES": {"country": "Spain", "languages": ["es", "en"], "file": "ep24_es_private.json"},
    "HU": {"country": "Hungary", "languages": ["hu", "en"], "file": "ep24_hu_private.json"},
    "HR": {"country": "Croatia", "languages": ["hr", "en"], "file": "ep24_hr_private.json"},
    "FR": {"country": "France", "languages": ["fr", "en"], "file": "ep24_fr_private.json"},
    "BG": {"country": "Bulgaria", "languages": ["bg", "en"], "file": "ep24_bg_private.json"},
}
LAYER_ORDER = {"common": 0, "eu": 1, "country": 2, "language": 3, "researcher": 4}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def identity_key(text: Any) -> str:
    """Conservative identity key for codebook labels and aliases.

    Deliberately identical in behaviour to ``roihu_memory.surface_key``: Unicode
    NFC, casefold, and internal-whitespace collapse, and nothing else. Accents,
    qualifiers and punctuation are preserved, because those distinctions are
    meaningful for discourse analysis and must not be normalized away.

    Using the same key on both sides is what lets a codebook entry and a memory
    object agree on identity. A bare ``str.casefold()`` does not: it leaves
    leading/trailing whitespace and NBSP untouched, so the same concept written
    with a stray space would hash to a different ``CB-`` id than memory gives it,
    and would also miss the merge's duplicate detection.

    This normalizes the KEY, never the stored label: the original text is kept
    verbatim in the entry.
    """
    return " ".join(unicodedata.normalize("NFC", str(text or "")).strip().casefold().split())


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.casefold()) if len(token) > 2}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceRef:
    source_type: str = ""
    url: str = ""
    title: str = ""
    language: str = ""
    publication_date: str = ""
    retrieval_date: str = ""
    evidence_locator: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodebookEntry:
    entry_id: str
    kind: str
    label: str
    aliases: list[str] = field(default_factory=list)
    country: str = ""
    source_languages: list[str] = field(default_factory=list)
    english_label: str = ""
    definition: str = ""
    english_definition: str = ""
    disambiguation: str = ""
    entity_type: str = ""
    review_state: str = "PROVISIONAL"
    origin: str = "public_context"
    locked: bool = False
    valid_from: str = ""
    valid_to: str = ""
    layer: str = "country"
    sources: list[SourceRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def forms(self) -> list[str]:
        values = [self.label, self.english_label, *self.aliases]
        return list(dict.fromkeys(v for v in (_clean(x) for x in values) if v))


def _entry_id(item: dict[str, Any], kind: str, label: str, country: str) -> str:
    explicit = _clean(item.get("canonical_id") or item.get("id") or item.get("entry_id"))
    if explicit:
        return explicit
    disambiguation = _clean(item.get("disambiguation"))
    # Same normalization as memory, so an id is stable under whitespace and
    # Unicode-form drift. An explicit id above is still returned untouched.
    seed = "|".join((country.upper(), kind, identity_key(label), identity_key(disambiguation)))
    return f"CB-{hashlib.sha256(seed.encode()).hexdigest()[:18]}"


def _source_ref(value: Any) -> SourceRef:
    if isinstance(value, str):
        return SourceRef(url=value if value.startswith("http") else "", evidence_locator="" if value.startswith("http") else value, raw={"value": value})
    if not isinstance(value, dict):
        return SourceRef(raw={"value": value})
    return SourceRef(
        source_type=_clean(value.get("source_type") or value.get("type") or value.get("origin")),
        url=_clean(value.get("url") or value.get("doi")),
        title=_clean(value.get("title")),
        language=_clean(value.get("language")),
        publication_date=_clean(value.get("publication_date") or value.get("date")),
        retrieval_date=_clean(value.get("retrieval_date") or value.get("retrieved_at")),
        evidence_locator=_clean(value.get("evidence_locator") or value.get("locator") or value.get("row")),
        raw=value,
    )


def _normalize_item(item: dict[str, Any], *, kind: str, default_country: str, default_language: str = "", layer: str = "country") -> CodebookEntry | None:
    label = _clean(item.get("label") or item.get("name") or item.get("canonical") or item.get("canonical_name"))
    if not label:
        return None
    aliases = item.get("aliases") or item.get("surface_forms") or item.get("variants") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    country = _clean(item.get("country_code") or item.get("country") or default_country).upper()
    source_lang = _clean(item.get("language") or default_language).lower()
    source_languages = item.get("source_languages") or ([source_lang] if source_lang else [])
    if isinstance(source_languages, str):
        source_languages = [source_languages]
    source_values = item.get("sources") or item.get("source_refs") or []
    if not source_values and item.get("provenance") is not None:
        source_values = [item.get("provenance")]
    if not isinstance(source_values, list):
        source_values = [source_values]
    origin = _clean(item.get("origin") or item.get("provenance_class") or item.get("status") or "public_context")
    reviewed = item.get("reviewed")
    review_state = _clean(item.get("review_state") or item.get("state"))
    if not review_state:
        review_state = "CANONICAL" if reviewed is True or origin in {"researcher_private", "researcher-grounded", "human"} else "PROVISIONAL"
    locked = bool(item.get("locked") or item.get("human_lock") or item.get("do_not_auto_change"))
    if origin in {"researcher_private", "researcher-grounded", "human"}:
        locked = True if item.get("locked") is None else locked
    return CodebookEntry(
        entry_id=_entry_id(item, kind, label, country),
        kind=kind,
        label=label,
        aliases=[_clean(v) for v in aliases if _clean(v)],
        country=country,
        source_languages=[_clean(v).lower() for v in source_languages if _clean(v)],
        english_label=_clean(item.get("english_label") or item.get("label_en") or item.get("english")),
        definition=_clean(item.get("definition") or item.get("description")),
        english_definition=_clean(item.get("english_definition") or item.get("definition_en")),
        disambiguation=_clean(item.get("disambiguation") or item.get("ambiguity_notes")),
        entity_type=_clean(item.get("entity_type") or item.get("type")),
        review_state=review_state.upper(), origin=origin, locked=locked,
        valid_from=_clean(item.get("valid_from")), valid_to=_clean(item.get("valid_to")), layer=layer,
        sources=[_source_ref(v) for v in source_values], metadata=dict(item.get("metadata") or {}),
    )


def load_codebook(path: str | Path, *, layer: str = "country") -> tuple[list[CodebookEntry], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    country = _clean(payload.get("country_code") or payload.get("country") or ("COMMON" if "common" in path.name else "")).upper()
    language = _clean(payload.get("language")).lower()
    candidates: list[tuple[str, dict[str, Any]]] = []
    for item in payload.get("entries", []) or []:
        if isinstance(item, dict):
            candidates.append((_clean(item.get("kind")) or "entity", item))
    for field_name, kind in (("entities", "entity"), ("themes", "topic"), ("topics", "topic"), ("signifiers", "signifier"), ("actors", "actor"), ("formations", "formation")):
        for item in payload.get(field_name, []) or []:
            candidates.append((kind, {"label": item} if isinstance(item, str) else item))
    entries = []
    for kind, item in candidates:
        if not isinstance(item, dict):
            continue
        if kind == "theme":
            kind = "topic"
        normalized = _normalize_item(item, kind=kind, default_country=country, default_language=language, layer=layer)
        if normalized:
            entries.append(normalized)
    meta = {"path": path.name, "sha256": _sha(path), "country": country, "language": language, "schema": _clean(payload.get("schema")), "entry_count": len(entries)}
    return entries, meta


# Researcher-note workbook columns (`research_notes.xlsx`) that researchers use
# to jot down candidate entities/persons/themes while doing the digital
# ethnography. These are controlled, human-authored seed columns, not finished
# codebook entries: every seed built from them below is PROVISIONAL and
# unlocked, so it must still be corroborated/reviewed, same as a model guess.
RESEARCH_NOTE_SEED_FIELDS: dict[str, tuple[str, ...]] = {
    "entity": ("new_entity", "researcher_new_persons"),
    "topic": ("new_theme", "researcher_new_themes"),
}
_SEED_SPLIT_RE = re.compile(r"[;\n]+")


def seed_entries_from_research_notes(
    rows: Iterable[dict[str, Any]],
    *,
    country: str,
    language: str = "",
    fields: dict[str, tuple[str, ...]] | None = None,
) -> list[CodebookEntry]:
    """Build PROVISIONAL codebook entries from free-text researcher-note seed columns.

    ``rows`` is any iterable of plain dicts (for example from
    ``research_notes.xlsx`` read into records) that may contain the columns
    named in ``fields`` (default: :data:`RESEARCH_NOTE_SEED_FIELDS`). A cell
    may hold more than one candidate label separated by ``;`` or a newline;
    each non-empty candidate becomes its own provisional entry so it can later
    be reviewed, merged with an existing canonical entry, or rejected. Nothing
    here is ever marked ``CANONICAL`` or ``locked``: researcher free text is a
    seed, not authoritative evidence, exactly like model-discovered guesses
    elsewhere in this module.
    """
    field_map = fields or RESEARCH_NOTE_SEED_FIELDS
    country_code = country.upper()
    seen: set[tuple[str, str]] = set()
    entries: list[CodebookEntry] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for kind, columns in field_map.items():
            for column in columns:
                raw_value = row.get(column)
                cell = _clean(raw_value)
                if not cell:
                    continue
                for candidate in _SEED_SPLIT_RE.split(cell):
                    label = _clean(candidate)
                    if not label:
                        continue
                    dedupe_key = (kind, identity_key(label))
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    entries.append(
                        CodebookEntry(
                            entry_id=_entry_id({}, kind, label, country_code),
                            kind=kind,
                            label=label,
                            country=country_code,
                            source_languages=[language.lower()] if language else [],
                            review_state="PROVISIONAL",
                            origin="research_notes_seed",
                            locked=False,
                            layer="researcher",
                            metadata={"source_field": column, "source_row_index": row_index},
                        )
                    )
    return entries


def private_root() -> Path:
    configured = os.getenv("LACLAUGPT_EP24_PRIVATE_ROOT") or os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT")
    if configured:
        return Path(configured)
    return Path("../LaclauGPT-Private/analysis/ep24")


def profile_paths(root: str | Path, country: str) -> list[tuple[Path, str]]:
    root = Path(root)
    codebooks = root / "codebooks"
    code = country.upper()
    if code not in COUNTRY_PROFILES:
        raise KeyError(f"unsupported EP24 country: {code}")
    return [
        (codebooks / "ep24_common_private.json", "common"),
        (codebooks / COUNTRY_PROFILES[code]["file"], "country"),
    ]


def english_label_required(entry: CodebookEntry) -> bool:
    """Whether an entry must carry an explicit English label.

    Policy: every entry sourced in a non-English language needs english_label.
    If the English form is identical (for example a person's name), store the
    identical string explicitly. Omission is allowed only with a non-empty
    metadata["english_label_exempt_reason"] so the exception is auditable.
    """
    langs = [lang for lang in entry.source_languages if lang]
    if not langs or not any(lang != "en" for lang in langs):
        return False
    return not bool(_clean(entry.metadata.get("english_label_exempt_reason")))


def english_label_coverage(entries: Iterable[CodebookEntry]) -> dict[str, Any]:
    """Return bilingual-label QA metrics without changing codebook content."""
    items = list(entries)
    required = [entry for entry in items if english_label_required(entry)]
    missing = [entry for entry in required if not entry.english_label]
    exempt = [
        entry for entry in items
        if _clean(entry.metadata.get("english_label_exempt_reason"))
        and entry.source_languages
        and any(lang != "en" for lang in entry.source_languages)
    ]
    present = len(required) - len(missing)
    return {
        "required_count": len(required),
        "present_count": present,
        "missing_count": len(missing),
        "missing_entry_ids": [entry.entry_id for entry in missing],
        "exempt_count": len(exempt),
        "coverage_pct": round((present / len(required) * 100), 1) if required else 100.0,
        "state": "PASS" if not missing else "REVIEW_REQUIRED",
    }


def load_profile(root: str | Path, country: str, *, language: str = "", strict_english: bool | None = None) -> tuple[list[CodebookEntry], dict[str, Any]]:
    loaded: list[tuple[list[CodebookEntry], dict[str, Any], str]] = []
    for path, layer in profile_paths(root, country):
        if path.exists():
            entries, meta = load_codebook(path, layer=layer)
            loaded.append((entries, meta, layer))
    if not loaded:
        raise FileNotFoundError(f"no private EP24 codebooks found for {country} under {Path(root) / 'codebooks'}")
    merged: dict[tuple[str, str, str], CodebookEntry] = {}
    conflicts: list[dict[str, Any]] = []
    for entries, _meta, _layer in loaded:
        for entry in entries:
            key = (entry.kind, identity_key(entry.label), entry.country or country.upper())
            old = merged.get(key)
            if old is None:
                merged[key] = entry
                continue
            if old.locked and not entry.locked:
                conflicts.append({"kept": old.entry_id, "rejected": entry.entry_id, "reason": "human_lock"})
                continue
            if entry.locked and not old.locked:
                conflicts.append({"kept": entry.entry_id, "rejected": old.entry_id, "reason": "human_lock"})
                merged[key] = entry
                continue
            if old.locked and entry.locked and asdict(old) != asdict(entry):
                conflicts.append({"kept": old.entry_id, "rejected": entry.entry_id, "reason": "locked_conflict_needs_review"})
                continue
            if LAYER_ORDER.get(entry.layer, 0) >= LAYER_ORDER.get(old.layer, 0):
                merged[key] = entry
    entries = list(merged.values())
    aliases: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        for form in entry.forms:
            aliases.setdefault((entry.kind, identity_key(form)), set()).add(entry.entry_id)
    ambiguous = sorted({form for (_kind, form), ids in aliases.items() if len(ids) > 1})
    fingerprint = hashlib.sha256("|".join(sorted(m[1]["sha256"] for m in loaded)).encode()).hexdigest()
    english_qa = english_label_coverage(entries)
    missing_english = english_qa["missing_entry_ids"]
    if strict_english is None:
        strict_english = os.getenv("LACLAUGPT_CODEBOOK_ENGLISH_STRICT", "").strip().casefold() in {"1", "true", "yes", "on"}
    if strict_english and english_qa["missing_count"]:
        raise ValueError(
            f"{country.upper()} codebook requires English-label review: "
            f"{english_qa['missing_count']}/{english_qa['required_count']} required entries are missing english_label"
        )
    sourced = [entry.entry_id for entry in entries if entry.sources]
    source_languages = sorted({
        source.language
        for entry in entries
        for source in entry.sources
        if source.language
    })
    return entries, {
        "country": country.upper(), "language": language.lower(), "fingerprint": fingerprint,
        "books": [m[1] for m in loaded], "entry_count": len(entries), "conflicts": conflicts,
        "ambiguous_forms": ambiguous, "missing_english_entry_ids": missing_english,
        "missing_english_count": len(missing_english), "english_label_coverage": english_qa,
        "qa_state": english_qa["state"], "sourced_entry_count": len(sourced),
        "source_languages": source_languages,
        "evidence_role": "background_context_not_source_evidence",
    }


def boundary_matches(form: str, query: str) -> bool:
    """True when ``form`` occurs in ``query`` as a whole token, not a substring.

    This is the **strict** matcher, and it exists because a plain
    ``form in query`` test is wrong for short party acronyms. Measured on this
    module: ``PiS`` (Poland's ruling party) matches the unrelated word
    ``Pisarz`` ("writer"), and ``HDZ`` matches ``HDZx``. A hand-corrected
    coverage count in the Croatia audit came from the same bug (``Možemo``
    matched ``Mozemohr``, ``SDSS`` matched ``Republika Srpska``).

    Use this for **identity and coverage** questions, where a false positive
    silently attributes an actor or inflates a number.

    Do **not** use it for the retrieval scorer: it deliberately rejects
    inflected forms, and inflection recall is wanted for the EP24 languages.
    ``ilmasto`` must still retrieve for the Finnish query ``ilmastosta``.
    See ``score_entry`` for that side of the trade and why the two differ.
    """
    needle = form.casefold().strip()
    if not needle:
        return False
    haystack = query.casefold()
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return False
        before = haystack[index - 1] if index else ""
        after_index = index + len(needle)
        after = haystack[after_index] if after_index < len(haystack) else ""
        if not _is_word_char(before) and not _is_word_char(after):
            return True
        start = index + 1


def _is_word_char(char: str) -> bool:
    """Whether ``char`` is part of a word, in any script.

    ``str.isalnum()`` is the bulk of it; underscore is included so that
    ``foo_bar`` does not read as a ``foo`` boundary, matching how Python's own
    ``\\w`` behaves in the regexes this module already uses.
    """
    return bool(char) and (char.isalnum() or char == "_")


def entity_type_distribution(entries: Iterable[CodebookEntry]) -> dict[str, int]:
    """Count entries per ``entity_type`` (falling back to ``kind``).

    The Croatia audit found "zero parties" in the legacy entity layer only by
    looking at a type breakdown; in aggregate output the gap was invisible,
    because 41 distinct entities all looked populated. Any codebook review
    should print this.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        key = (entry.entity_type or entry.kind or "unknown").casefold() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def score_entry(query: str, entry: CodebookEntry) -> float:
    """Lexical retrieval score: recall-oriented, so it matches substrings.

    Retrieval and identity are different questions, and they need opposite
    biases:

    - **Retrieval** (this function) feeds background context, which is advisory.
      A miss costs context; the EP24 languages are heavily inflective, so
      ``ilmasto`` must retrieve for ``ilmastosta`` and a substring match is what
      buys that. Over-matching is a quality problem, bounded by the selection
      limit and by the evidence firewall, not a correctness one.
    - **Identity** (``roihu_identity.resolve_surface``) decides what an actor
      *is*. A false positive there silently attributes a statement to the wrong
      party. That path uses exact ``identity_key`` matching only, and abstains
      otherwise. ``boundary_matches`` belongs to that side and to coverage
      counting.

    The substring behaviour here is therefore deliberate and load-bearing, not
    an oversight: ``test_private_profile_merge_keeps_locked_human_entry_and_bilingual_context``
    depends on the Finnish inflection case.
    """
    q = query.casefold()
    if not q.strip():
        return 0.0
    for form in entry.forms:
        normalized = form.casefold().strip()
        if not normalized:
            continue
        if len(normalized) >= 3 and normalized in q:
            return 1.0
        if len(normalized) < 3 and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", q):
            return 1.0
    q_tokens = _tokens(query)
    entry_tokens = set()
    for value in [*entry.forms, entry.definition, entry.english_definition]:
        entry_tokens.update(_tokens(value))
    return len(q_tokens & entry_tokens) / max(1, len(entry_tokens))


def select_context(query: str, entries: Iterable[CodebookEntry], *, country: str, language: str = "", limit: int = 8, threshold: float = 0.15) -> tuple[list[CodebookEntry], dict[str, Any]]:
    scoped = [entry for entry in entries if entry.country in {"", "COMMON", country.upper()}]
    ranked = sorted(((score_entry(query, e), e) for e in scoped), key=lambda pair: (-pair[0], pair[1].kind, pair[1].label.casefold()))
    selected = [(score, entry) for score, entry in ranked[: max(0, limit)] if score >= threshold]
    return [e for _, e in selected], {
        "country": country.upper(), "language": language.lower(), "limit": limit, "threshold": threshold,
        "selection_method": "deterministic_lexical_v2_bilingual", "evidence_role": "background_context_not_source_evidence",
        "selected": [{"entry_id": e.entry_id, "kind": e.kind, "label": e.label, "english_label": e.english_label, "score": round(score, 6)} for score, e in selected],
    }


def context_block(query: str, entries: Iterable[CodebookEntry], *, country: str, language: str = "", limit: int = 8, threshold: float = 0.15) -> tuple[str, dict[str, Any]]:
    selected, provenance = select_context(query, entries, country=country, language=language, limit=limit, threshold=threshold)
    if not selected:
        return "", provenance
    lines = ["[EP24 CODEBOOK CONTEXT] Background context only, not evidence from the current item and not proof of an actor's beliefs or the author's agreement."]
    for entry in selected:
        bilingual = entry.label if not entry.english_label or entry.english_label == entry.label else f"{entry.label} / {entry.english_label}"
        definition = entry.english_definition or entry.definition
        lines.append(f"- {entry.kind}: {bilingual}" + (f" | {definition}" if definition else ""))
    return "\n".join(lines), provenance


def coverage_manifest() -> list[dict[str, Any]]:
    return [
        {
            "country_code": code,
            "country": meta["country"],
            "languages": meta["languages"],
            "private_file": meta["file"],
            "english_required": True,
            "english_gap_policy": "report_missing_never_silent_country_fallback",
        }
        for code, meta in COUNTRY_PROFILES.items()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", default=str(private_root()))
    parser.add_argument("--country", default="FI")
    parser.add_argument("--language", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest:
        print(json.dumps(coverage_manifest(), ensure_ascii=False, indent=2))
        return 0
    entries, meta = load_profile(args.private_root, args.country, language=args.language)
    block, selection = context_block(args.text, entries, country=args.country, language=args.language)
    print(json.dumps({"profile": meta, "selection": selection, "context": block}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
