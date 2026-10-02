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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

    This function is also the **id seed** (``_entry_id``) and
    ``roihu_memory.stable_id``, so its output must not change: doing so would
    move every stored id. See ``elision_key`` for the lossy fold that is applied
    where a lossy comparison is safe (merge/dedupe) but an id must not move.
    """
    return " ".join(unicodedata.normalize("NFC", str(text or "")).strip().casefold().split())


#: Apostrophe-family characters that all mark the SAME French/Italian elision.
#: The difference between them is which keyboard or publishing pipeline produced
#: the text, not what the text means. Source of truth is
#: ``scripts/ep24/apostrophe_hygiene.py::SAFE_ELISION``; this mirror exists so the
#: identity layer does not have to import a script module at runtime.
ELISION_APOSTROPHES = frozenset({"\u0027", "\u2019", "\u2018", "\u02bc"})

#: The apostrophe this project writes into its own labels (ASCII), matching
#: ``scripts/ep24/apostrophe_hygiene.py::PROJECT_APOSTROPHE``.
PROJECT_APOSTROPHE = "\u0027"

#: Explicitly NOT folded. Kept as a named constant so both the fold and the
#: tests can assert the boundary rather than relying on the loop's default.
NOT_ELISION: frozenset = frozenset(
    {
        "\u2032",  # PRIME — minutes/feet, not an apostrophe
        "\u2033",  # DOUBLE PRIME
        "\u0060",  # GRAVE ACCENT — not an apostrophe at all
        "\u201c", "\u201d",  # left/right double quotation marks
        "\u002d", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212",  # dash family
        "\u2026",  # ellipsis
    }
)


def elision_key(text: Any) -> str:
    """Identity key with apostrophe-elision variants folded. **Lossy, by design.**

    ``identity_key`` preserves punctuation, correctly — an apostrophe is
    analytically meaningful and stripping punctuation was the false-merge defect
    that ``tests/test_identity_normalization.py`` pins against. But French and
    Italian elision is written with either the ASCII apostrophe (keyboards, ASR)
    or U+2019 (word processors, iOS autocorrect, much news web output), and the
    two are canonically unrelated (``Po`` vs ``Pf``), so NFC cannot unify them:

        identity_key("Besoin d'Europe") != identity_key("Besoin d\u2019Europe")

    That makes one entity two identities — two ``CB-`` ids, two memory objects,
    both CANONICAL. See ``docs/ep24_country_audits/FR.md`` Finding 4 and #102.

    **Where this may be used:** comparison only — merge/dedupe keys, duplicate
    detection, and "are these two labels the same entity". Folding here can only
    ever merge two entries that already share an ``identity_key`` apart from the
    apostrophe.

    **Where it may NOT be used:** anywhere the result is stored or hashed into an
    id. ``identity_key`` is the id seed, so folding inside it would move every
    existing entry id whose label carries a typographic apostrophe. That is the
    migration problem #102 asks to solve deliberately, not a side effect to
    introduce.

    The fold is narrow on purpose. Only the elision family is folded; primes,
    grave accents, the dash family and quotation marks are left untouched,
    because unifying those would be a semantic change rather than a
    normalisation (``Most`` vs ``Most`` is a party; a prime is minutes).
    """
    return identity_key("".join(PROJECT_APOSTROPHE if c in ELISION_APOSTROPHES else c for c in str(text or "")))


def elision_merged(a: str, b: str) -> bool:
    """True when two labels are one entity *only because* of the elision fold.

    This is the crosswalk predicate: ``True`` means ``identity_key`` keeps them
    apart while ``elision_key`` unifies them, i.e. exactly the class of
    apostrophe-only split this change resolves. Used to record merged pairs
    explicitly rather than merging them silently.
    """
    return identity_key(a) != identity_key(b) and elision_key(a) == elision_key(b)


def _elision_record(kept: CodebookEntry, folded: CodebookEntry) -> dict[str, Any]:
    """Crosswalk row for one apostrophe-only merge (#102).

    Records both the labels AND the ids, because ids are what the migration
    question is about: ``kept_id`` survives, ``folded_id`` was the id that the
    other spelling would have produced. A run against a live book therefore
    yields the exact list of ids that will stop being generated, which is what
    the acceptance criterion "migration/compatibility behavior for existing IDs
    is documented and tested" needs to be checkable rather than described.
    """
    return {
        "kept": kept.label,
        "folded": folded.label,
        "kept_id": kept.entry_id,
        "folded_id": folded.entry_id,
        "kept_layer": kept.layer,
        "folded_layer": folded.layer,
        "reason": "elision_apostrophe",
    }


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
    # Public-context codebook entries nest their authoritative URLs inside
    # ``provenance.sources`` (a list of URL strings), not at entry level. Read
    # them explicitly: passing the bare provenance dict to ``_source_ref``
    # yields a single ref whose ``url`` is empty, so every nested URL used to be
    # dropped silently and ``entry.sources[].url`` was always "".
    nested_values: list[Any] = []
    for value in source_values:
        if not isinstance(value, dict):
            continue
        for key in ("sources", "source_refs", "urls"):
            inner = value.get(key)
            if isinstance(inner, str):
                inner = [inner]
            if isinstance(inner, list):
                nested_values.extend(inner)
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
        sources=[_source_ref(v) for v in [*source_values, *nested_values]], metadata=dict(item.get("metadata") or {}),
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
    "entity": ("entities",),
    "topic": ("themes",),
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
                try:
                    decoded = json.loads(cell)
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded = None
                if isinstance(decoded, list):
                    candidates = [str(item).strip() for item in decoded if str(item).strip()]
                else:
                    candidates = _SEED_SPLIT_RE.split(cell)
                for candidate in candidates:
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


def _has_language_metadata(entry: CodebookEntry) -> bool:
    """Whether the entry states any language at all.

    The ``common`` codebook layer records no ``language`` at file or entry level,
    so its entries arrive with an empty ``source_languages``. Treating that as
    "nothing to translate" is how 2694 entries per country became invisible.
    """
    return any(lang for lang in entry.source_languages)


_NON_ENGLISH_MARKERS = re.compile(
    r"\b(partia|partido|partei|parti|stranka|puolue|koalicja|koalicija|frente|blok|"
    r"allianssi|rassemblement|democracia|demokratie|demokratia|zieloni|zielone|"
    r"verdes|gr[uü]ne|obywatelska|solidarna|suwerenna|moderaterna|"
    r"socialdemokraterna|sverigedemokraterna|v[aä]nsterpartiet|milj[oö]partiet|"
    r"liberalerna|vasemmistoliitto|kokoomus|keskusta|perussuomalaiset|fidesz|"
    r"rassemblement|r[eé]publicains|fianna|sinn|fein|partidos?|coalici[oó]n)\b",
    re.IGNORECASE,
)


def _looks_like_person_name(label: str) -> bool:
    """A plain personal name: 2–4 capitalised words.

    Exempt from the English requirement because the English form is normally the
    same string (``Pedro Sánchez``); demanding a separate label there manufactures
    work. Organisation words are excluded by the caller so that ``Les
    Républicains`` and ``Fianna Fáil`` are still required.
    """
    text = _clean(label)
    if not text or any(ch.isdigit() for ch in text):
        return False
    words = [w for w in text.split() if w]
    if not (2 <= len(words) <= 4):
        return False
    return all(w[0].isupper() for w in words if w[:1].isalpha())


def _looks_like_identifier(label: str) -> bool:
    """Handles and URLs: language-neutral identifiers needing no gloss."""
    text = _clean(label)
    return text.startswith(("@", "http://", "https://"))


def label_looks_english(label: str) -> bool:
    """Whether a canonical label is plausibly already English.

    Conservative: diacritics or a recognisable non-English political word mark
    a label as *not* English. Over-reporting costs one redundant gloss;
    under-reporting costs a silently missing translation, so the bias is
    deliberate. Known limitations are documented in
    ``docs/EP24_BILINGUAL_CODEBOOK_POLICY.md``.
    """
    text = _clean(label)
    if not text:
        return True
    if any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", text)):
        return False
    return not _NON_ENGLISH_MARKERS.search(text)


# Organisation markers used only by the translation-work metric. They prevent
# capitalised organisation labels from being mistaken for personal names.
_ORG_MARKERS = re.compile(
    r"\b(partia|partido|partei|parti|stranka|puolue|koalicja|koalicija|frente|blok|"
    r"front|alliance|allianssi|rassemblement|party|parties|movement|union|liga|liitto|"
    r"ryhmä|verdes|grüne|zieloni|zielone|moderaterna|socialdemokraterna|"
    r"sverigedemokraterna|vänsterpartiet|miljöpartiet|liberalerna|fianna|sinn|fine|"
    r"les|républicains|republikaner|sozialdemokraten)\b",
    re.IGNORECASE,
)


def english_translation_required(entry: CodebookEntry) -> bool:
    """Whether the canonical label needs an English translation/gloss.

    This metric answers a narrower question than `english_label_required`:
    does a researcher need to supply a distinct English rendering? Personal
    names and labels that are already English do not count as translation work.
    Handles and URLs are language-neutral. Organisation labels are not exempted
    merely because they resemble a capitalised personal name.
    """
    label = _clean(entry.label)
    if not label or _looks_like_identifier(label):
        return False
    if _looks_like_person_name(label) and not _ORG_MARKERS.search(label):
        return False
    return not label_looks_english(label)


# Backwards-compatible API name from #110. Kept as a delegating alias so callers
# do not break, but the canonical name states the question it answers.
def entry_needs_english_label(entry: CodebookEntry) -> bool:
    return english_translation_required(entry)


def english_label_required(entry: CodebookEntry) -> bool:
    """Whether an entry must carry an explicit English label.

    Policy: every entry sourced in a non-English language needs english_label.
    If the English form is identical (for example a person's name), store the
    identical string explicitly. Omission is allowed only with a non-empty
    metadata["english_label_exempt_reason"] so the exception is auditable.

    Entries with **no language metadata** are decided on their label instead of
    being skipped. The ``common`` layer carries no ``language``, so an empty
    ``source_languages`` means "unknown" — and treating unknown as "already
    English" exempted 2694 entries per country *by omission*: not present, not
    absent, not exempt, simply uncounted. An entry with an unknown language is
    required when its label is not already English, so the gap is visible:

    * already-English label (``Abortion``) -> not required, nothing to translate;
    * non-English label (``Rassemblement National``) -> required;
    * person name (``Pedro Sánchez``) -> not required, the form is identical;
    * handle or URL -> not required, language-neutral.
    """
    if _clean(entry.metadata.get("english_label_exempt_reason")):
        return False
    if not _has_language_metadata(entry):
        label = _clean(entry.label)
        if not label or _looks_like_identifier(label):
            return False
        if _looks_like_person_name(label) and not _NON_ENGLISH_MARKERS.search(label):
            return False
        return not label_looks_english(label)
    return any(lang != "en" for lang in entry.source_languages if lang)


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
    # Crosswalk for #102: pairs that merged ONLY because of the elision fold.
    # `identity_key` still keeps them apart (so no id moves), but the merge is
    # the place where a lossy comparison is safe, so the two spellings collapse
    # to one entry here instead of becoming two canonical objects. Recorded
    # explicitly so the merge is auditable rather than silent.
    elision_merges: list[dict[str, Any]] = []
    for entries, _meta, _layer in loaded:
        for entry in entries:
            key = (entry.kind, elision_key(entry.label), entry.country or country.upper())
            old = merged.get(key)
            if old is None:
                merged[key] = entry
                continue
            if old.locked and not entry.locked:
                conflicts.append({"kept": old.entry_id, "rejected": entry.entry_id, "reason": "human_lock"})
                if elision_merged(old.label, entry.label):
                    elision_merges.append(_elision_record(old, entry))
                continue
            if entry.locked and not old.locked:
                conflicts.append({"kept": entry.entry_id, "rejected": old.entry_id, "reason": "human_lock"})
                if elision_merged(old.label, entry.label):
                    elision_merges.append(_elision_record(entry, old))
                merged[key] = entry
                continue
            if old.locked and entry.locked and asdict(old) != asdict(entry):
                conflicts.append({"kept": old.entry_id, "rejected": entry.entry_id, "reason": "locked_conflict_needs_review"})
                continue
            if LAYER_ORDER.get(entry.layer, 0) >= LAYER_ORDER.get(old.layer, 0):
                if elision_merged(old.label, entry.label):
                    elision_merges.append(_elision_record(entry, old))
                merged[key] = entry
            elif elision_merged(old.label, entry.label):
                elision_merges.append(_elision_record(old, entry))
    entries = list(merged.values())
    aliases: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        for form in entry.forms:
            aliases.setdefault((entry.kind, identity_key(form)), set()).add(entry.entry_id)
    ambiguous = sorted({form for (_kind, form), ids in aliases.items() if len(ids) > 1})
    fingerprint = hashlib.sha256("|".join(sorted(m[1]["sha256"] for m in loaded)).encode()).hexdigest()
    # Two computations, deliberately kept separate:
    #
    # * `english_qa` is the language-gated QA block (#108). It keeps the
    #   audit surface (`required/present/missing/exempt`, `state`) that the
    #   strict gate and its report expose.
    # * `missing_english` is the *policy* predicate (#101/#110). The gated
    #   version cannot see the shared `common` layer at all, because that layer
    #   carries no language metadata -- so the gate under-reports the real gap
    #   and can flag an already-English label instead. The policy predicate is
    #   therefore the authority for which entry ids are missing.
    #
    # On a country book whose gaps are all in the language-tagged layers the two
    # agree on the count. They disagree on identity exactly where it matters, so
    # both are kept rather than collapsing one into the other.
    english_qa = english_label_coverage(entries)
    missing_english = [
        entry.entry_id
        for entry in entries
        if english_translation_required(entry) and not entry.english_label
    ]
    if strict_english is None:
        strict_english = os.getenv("LACLAUGPT_CODEBOOK_ENGLISH_STRICT", "").strip().casefold() in {"1", "true", "yes", "on"}
    if strict_english and missing_english:
        raise ValueError(
            f"{country.upper()} codebook requires English-label review: "
            f"{len(missing_english)}/{len(entries)} entries are missing english_label"
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
        # #102: apostrophe-only merges, recorded so the fold is auditable rather
        # than silent. Empty when no label carries a typographic apostrophe.
        "elision_merges": sorted(elision_merges, key=lambda m: (m["kept"], m["folded"])),
        "elision_merge_count": len({(m["kept"], m["folded"]) for m in elision_merges}),
        "evidence_role": "background_context_not_source_evidence",
    }


def boundary_matches(form: str, query: str) -> bool:
    """True when ``form`` occurs in ``query`` as a whole token, not a substring.

    This is the **strict** matcher, and it exists because a plain
    ``form in query`` test is wrong for short party acronyms. Measured on this
    module: ``PiS`` (Poland's ruling party) matches the unrelated word
    ``Pisarz`` ("writer"), ``HDZ`` matches ``HDZx``, and ``Most`` (a Croatian
    party) matches ``Mostar``.

    A hand-corrected coverage count in the Croatia audit came from the same
    class of bug. Note the correction on that example (Croatia pass 3): the
    audit's ``Možemo``/``Mozemohr`` pair reproduces only through the
    *auditor's* diacritic-stripping fold, **not** through this runtime matcher,
    which preserves diacritics; and its ``SDSS``/``Republika Srpska`` pair does
    not reproduce at all. ``Most``/``Mostar`` and ``HDZ``/``HDZx`` are the pairs
    that reproduce cleanly against this function.

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


def _reviewed_short_form(entry: CodebookEntry, form: str) -> bool:
    """Whether a short form is explicit/reviewed enough for retrieval.

    Aliases stored in a codebook are explicit declarations and are safe to use
    with the short-form boundary rule. A short canonical label that is merely
    provisional is not promoted into retrieval unless the entry is reviewed,
    locked, or the form is named in ``metadata.reviewed_short_aliases``.
    """
    wanted = identity_key(form)
    if any(identity_key(alias) == wanted for alias in entry.aliases):
        return True
    if entry.locked or entry.review_state.upper() == "CANONICAL":
        return True
    explicit = entry.metadata.get("reviewed_short_aliases") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    return any(identity_key(value) == wanted for value in explicit)


def _short_scope_allows(
    entry: CodebookEntry,
    form: str,
    *,
    language: str = "",
    election: str = "",
) -> bool:
    """Apply optional alias-level language/election constraints.

    Country scope is applied before scoring and entity kind remains part of the
    entry identity. Codebooks that need tighter context may additionally define
    ``metadata.short_alias_languages`` and/or
    ``metadata.short_alias_elections`` as maps keyed by alias.
    """
    wanted = identity_key(form)
    for key, current in (
        ("short_alias_languages", language.lower()),
        ("short_alias_elections", election.casefold()),
    ):
        if not current:
            continue
        mapping = entry.metadata.get(key) or {}
        if not isinstance(mapping, dict):
            continue
        allowed = None
        for alias, values in mapping.items():
            if identity_key(alias) == wanted:
                allowed = values
                break
        if allowed is None:
            continue
        if isinstance(allowed, str):
            allowed = [allowed]
        if current not in {str(value).casefold() for value in allowed}:
            return False
    return True


def _matching_short_forms(
    query: str,
    entry: CodebookEntry,
    *,
    language: str = "",
    election: str = "",
) -> set[str]:
    """Return reviewed, in-scope short forms that match query by boundaries."""
    out: set[str] = set()
    for form in entry.forms:
        normalized = form.casefold().strip()
        if not normalized or len(normalized) >= 5:
            continue
        if not _reviewed_short_form(entry, normalized):
            continue
        if not _short_scope_allows(entry, normalized, language=language, election=election):
            continue
        if boundary_matches(normalized, query):
            out.add(identity_key(normalized))
    return out


def score_entry(
    query: str,
    entry: CodebookEntry,
    *,
    language: str = "",
    election: str = "",
    blocked_short_forms: set[str] | frozenset[str] = frozenset(),
) -> float:
    """Lexical retrieval score with safe, scoped short-form matching.

    Forms shorter than five characters use whole-token matching. Declared
    aliases are accepted as explicit codebook forms; provisional short labels
    require review/lock metadata. Optional language/election constraints can
    further scope aliases. Ambiguous one-letter aliases may be blocked by the
    selector. Forms of five characters or more keep substring matching for
    inflection-friendly recall.
    """
    q = query.casefold()
    if not q.strip():
        return 0.0
    for form in entry.forms:
        normalized = form.casefold().strip()
        if not normalized:
            continue
        if len(normalized) < 5:
            key = identity_key(normalized)
            if key in blocked_short_forms:
                continue
            if not _reviewed_short_form(entry, normalized):
                continue
            if not _short_scope_allows(
                entry,
                normalized,
                language=language,
                election=election,
            ):
                continue
            if boundary_matches(normalized, q):
                return 1.0
            continue
        if normalized in q:
            return 1.0
    q_tokens = _tokens(query)
    entry_tokens = set()
    for value in [*entry.forms, entry.definition, entry.english_definition]:
        if len(value.casefold().strip()) >= 5 or value in {entry.definition, entry.english_definition}:
            entry_tokens.update(_tokens(value))
    return len(q_tokens & entry_tokens) / max(1, len(entry_tokens))


def select_context(
    query: str,
    entries: Iterable[CodebookEntry],
    *,
    country: str,
    language: str = "",
    election: str = "",
    limit: int = 8,
    threshold: float = 0.15,
) -> tuple[list[CodebookEntry], dict[str, Any]]:
    scoped = [entry for entry in entries if entry.country in {"", "COMMON", country.upper()}]

    short_matches: dict[str, set[str]] = {}
    for entry in scoped:
        for form in _matching_short_forms(query, entry, language=language, election=election):
            short_matches.setdefault(form, set()).add(entry.entry_id)
    ambiguous_short_forms = {
        form for form, entry_ids in short_matches.items() if len(entry_ids) > 1
    }
    ambiguous_one_letter_forms = {
        form for form in ambiguous_short_forms if len(form) == 1
    }

    ranked = sorted(
        (
            (
                score_entry(
                    query,
                    entry,
                    language=language,
                    election=election,
                    blocked_short_forms=ambiguous_one_letter_forms,
                ),
                entry,
            )
            for entry in scoped
        ),
        key=lambda pair: (-pair[0], pair[1].kind, pair[1].label.casefold()),
    )
    selected = [(score, entry) for score, entry in ranked[: max(0, limit)] if score >= threshold]
    return [entry for _, entry in selected], {
        "country": country.upper(),
        "language": language.lower(),
        "election": election,
        "limit": limit,
        "threshold": threshold,
        "selection_method": "deterministic_lexical_v5_scoped_short_alias_ambiguity",
        "evidence_role": "background_context_not_source_evidence",
        "ambiguous_short_forms": sorted(ambiguous_short_forms),
        "selected": [
            {
                "entry_id": entry.entry_id,
                "kind": entry.kind,
                "label": entry.label,
                "english_label": entry.english_label,
                "score": round(score, 6),
            }
            for score, entry in selected
        ],
    }

def context_block(
    query: str,
    entries: Iterable[CodebookEntry],
    *,
    country: str,
    language: str = "",
    election: str = "",
    limit: int = 8,
    threshold: float = 0.15,
) -> tuple[str, dict[str, Any]]:
    selected, provenance = select_context(
        query,
        entries,
        country=country,
        language=language,
        election=election,
        limit=limit,
        threshold=threshold,
    )
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


def english_translation_coverage_report(root: str | Path) -> dict[str, Any]:
    """Coverage of ``english_label`` for every EP24 country.

    Reports the *true* gap (entries whose label is not already English and which
    carry no ``english_label``) rather than a language-gated subset, so the
    ``common`` layer cannot hide from the number. Read-only.
    """
    per_country: list[dict[str, Any]] = []
    for code in sorted(COUNTRY_PROFILES):
        entries, meta = load_profile(root, code)
        needs = [e for e in entries if entry_needs_english_label(e)]
        missing = [e for e in needs if not e.english_label]
        by_layer: dict[str, int] = {}
        for entry in missing:
            by_layer[entry.layer] = by_layer.get(entry.layer, 0) + 1
        per_country.append({
            "country_code": code,
            "entries": len(entries),
            "needs_english": len(needs),
            "missing_english": len(missing),
            "missing_pct": round(len(missing) / len(entries) * 100, 1) if entries else 0.0,
            "missing_by_layer": by_layer,
            "missing_entry_ids": [e.entry_id for e in missing],
        })
    totals: dict[str, Any] = {
        "entries": sum(c["entries"] for c in per_country),
        "needs_english": sum(c["needs_english"] for c in per_country),
        "missing_english": sum(c["missing_english"] for c in per_country),
    }
    totals["missing_pct"] = (
        round(totals["missing_english"] / totals["entries"] * 100, 1) if totals["entries"] else 0.0
    )
    return {
        "kind": "ep24.english_translation_coverage/1",
        "policy": "translation work: english_label needed only when the canonical label needs an English rendering",
        "countries": per_country,
        "totals": totals,
    }


def assert_english_translation_coverage(
    report: dict[str, Any], *, max_missing_pct: float = 100.0
) -> None:
    """Raise when a country's bilingual gap exceeds a threshold.

    Exists so a large coverage gap cannot pass silently (issue #101). The
    default threshold permits any gap, which keeps the research pipeline running;
    QA should call this with an explicit ``max_missing_pct`` once labels are
    repaired.
    """
    offenders = [
        c for c in report.get("countries", []) if c["missing_pct"] > max_missing_pct
    ]
    if offenders:
        detail = ", ".join(f"{c['country_code']}={c['missing_pct']}%" for c in offenders)
        raise ValueError(
            f"bilingual coverage below threshold {max_missing_pct}% for: {detail}"
        )



# Compatibility wrappers for the #110 CLI/API names. These delegate to the
# translation-work metric and are intentionally not a second policy.
def bilingual_coverage_report(root: str | Path) -> dict[str, Any]:
    """Compatibility view of the translation-work report from #110."""
    report = english_translation_coverage_report(root)
    return {**report, "kind": "ep24.bilingual_coverage/1"}


def assert_bilingual_coverage(
    report: dict[str, Any], *, max_missing_pct: float = 100.0
) -> None:
    assert_english_translation_coverage(report, max_missing_pct=max_missing_pct)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", default=str(private_root()))
    parser.add_argument("--country", default="FI")
    parser.add_argument("--language", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--bilingual-report", action="store_true",
                        help="report english_label coverage for every EP24 country")
    parser.add_argument("--max-missing-pct", type=float, default=100.0,
                        help="with --bilingual-report: fail if any country exceeds this gap")
    args = parser.parse_args(argv)
    if args.bilingual_report:
        report = english_translation_coverage_report(args.private_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        assert_english_translation_coverage(report, max_missing_pct=args.max_missing_pct)
        return 0
    if args.manifest:
        print(json.dumps(coverage_manifest(), ensure_ascii=False, indent=2))
        return 0
    entries, meta = load_profile(args.private_root, args.country, language=args.language)
    block, selection = context_block(args.text, entries, country=args.country, language=args.language)
    print(json.dumps({"profile": meta, "selection": selection, "context": block}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
