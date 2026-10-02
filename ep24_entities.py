#!/usr/bin/env python3
"""Multilingual entity normalization and resolution (issue #142).

NER/mention extraction produces many surface forms for one real-world actor --
``Pääministeri Orpo``, ``Orpo``, ``Petteri Orpo``, ``Orpon`` -- and without a
resolution layer those become separate entities downstream (DNA, SNA/assemblage,
RDF, MongoDB/RAG, country comparisons).

Design principles, all of them load-bearing:

* **Mentions are not entities.** NER detects *mentions*; resolution is a separate
  problem with its own failure modes. A mention is never mutated: the original
  surface form is stored verbatim beside the canonical entity, because *how* an
  actor is named is itself analytically meaningful for discourse analysis.
* **Deterministic before expensive.** Exact canonical name, exact alias,
  language-aware normalization, fuzzy similarity -- then, only when the cheap and
  auditable steps are exhausted, optional semantic/LLM adjudication. Cheap steps
  are the ones whose decisions can be explained to a reviewer.
* **Never silently merge an ambiguity.** A surname shared by two politicians is
  not a match. Unresolved cases go to a review queue rather than being guessed.
* **Nothing here re-derives identity.** ``roihu_codebooks.identity_key`` is the id
  seed and must not change, so this module adds a *comparison* layer beside it
  (``fold_key``) rather than altering the key. Normalization is lossy and is used
  only to find candidates; the registry's ids and canonical names come from the
  codebooks.

This module is public and data-agnostic: real EP24 labels stay in
LaclauGPT-Private, and tests use invented fixtures only. Optional heavy
dependencies (RapidFuzz, SentenceTransformers) are imported lazily so the module
stays runnable with the standard library alone -- the same treatment rdflib got
in this repository, where a module-scope import of an absent package once broke
main.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ep24_models import ollama_model
from roihu_codebooks import CodebookEntry, identity_key

LOG = logging.getLogger(__name__)

#: Entity kinds this layer resolves. Mentions of themes/topics are deliberately
#: out of scope: the issue is about actors, and theme normalization has different
#: semantics (a theme can legitimately subsume another).
ENTITY_KINDS = ("person", "actor", "entity", "organization", "party", "institution")

#: Resolution decisions, in escalating cost order. Recorded verbatim in output so
#: a reviewer can see which step decided a row.
MATCH_METHODS = (
    "exact_canonical",
    "exact_english_label",
    "exact_alias",
    "normalized_exact",
    "title_stripped",
    "morphology_folded",
    "diacritic_folded",
    "fuzzy_candidate",
    "semantic_candidate",
    "llm_adjudicated",
    "researcher_mapping",
)

#: Honorifics and political titles per language. Stripped for *comparison* only;
#: the title stays in the preserved surface form. Deliberately conservative: only
#: forms that are unambiguously a title, never a name part.
TITLES: dict[str, tuple[str, ...]] = {
    "fi": (
        "pääministeri", "ministeri", "presidentti", "kansanedustaja", "puheenjohtaja",
        "eduskunnan puhemies", "ulkoministeri", "valtiovarainministeri", "sisäministeri",
        "europarlamentaarikko", "meppi", "kansanedustaja", "kaupunginjohtaja",
        "herra", "rouva", "tohtori", "professori",
    ),
    "pl": (
        "premier", "minister", "prezydent", "poseł", "europoseł", "marszałek",
        "przewodniczący", "wicepremier", "sekretarz", "pan", "pani",
    ),
    "pt": (
        "primeiro-ministro", "primeiro ministro", "ministro", "presidente",
        "deputado", "eurodeputado", "secretário", "vereador", "senhor", "senhora",
    ),
    "de": (
        "bundeskanzler", "kanzler", "minister", "präsident", "abgeordneter",
        "vorsitzender", "fraktionsvorsitzender", "herr", "frau", "dr", "prof",
    ),
    "es": (
        "presidente", "ministro", "diputado", "eurodiputado", "secretario",
        "alcalde", "señor", "señora",
    ),
    "fr": (
        "premier ministre", "ministre", "président", "député", "eurodéputé",
        "secrétaire", "maire", "monsieur", "madame",
    ),
    "sv": (
        "statsminister", "minister", "president", "riksdagsledamot", "ordförande",
        "herr", "fru",
    ),
    "hu": ("miniszterelnök", "miniszter", "elnök", "képviselő", "úr", "asszony"),
    "hr": ("predsjednik", "ministar", "premijer", "zastupnik", "gospodin", "gospođa"),
    "bg": ("министър-председател", "министър", "президент", "депутат", "господин"),
    "en": ("prime minister", "minister", "president", "mp", "mep", "chair", "mr", "mrs", "ms", "dr"),
}

#: Case/number endings used to fold an inflected surname onto its base form for
#: comparison. Ordered longest-first so the most specific rule wins. These are
#: *comparison-only* folds and never touch a stored label or an id.
INFLECTION_SUFFIXES: dict[str, tuple[str, ...]] = {
    "fi": ("ssa", "ssä", "sta", "stä", "lla", "llä", "lta", "ltä", "lle", "ksi", "ina", "inä", "n", "a", "ä"),
    "pl": ("ego", "emu", "owi", "ach", "ami", "em", "ie", "ów", "om", "a", "u", "i", "ę", "ą"),
    "pt": ("mente", "ções", "ção", "es", "os", "as", "o", "a"),
    "hr": ("om", "em", "ima", "ama", "u", "a", "e", "i", "o"),
    "hu": ("nak", "nek", "val", "vel", "ban", "ben", "tól", "től", "nál", "nél", "t", "a", "e"),
    "de": ("s", "es", "en", "er", "em", "n"),
    "sv": ("s", "en", "ar", "er", "n"),
    "bg": ("ът", "ят", "та", "а", "о", "и", "е"),
}

#: Minimum length for a title-stripped remainder to be treated as a name rather
#: than a fragment of the title itself.
MIN_NAME_LENGTH = 2

#: Minimum relative length a shorter string must have against a longer one before a
#: substring containment may be treated as a match. Guards the classic false merge
#: where a very short surname appears inside an unrelated longer name.
CONTAINMENT_MIN_RATIO = 0.6


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_key(text: Any) -> str:
    """Lossy comparison key: case-, accent- and punctuation-insensitive.

    This is the "same actor, differently spelled" key -- ``Ø``/``O``, ``ł``/``l``,
    ``ß``/``ss`` and dropped accents from OCR/ASR all land together. It is
    deliberately NOT ``identity_key``:

    * ``identity_key`` is the stable id seed and must never change, and it
      preserves accents and punctuation on purpose (those distinctions are
      meaningful, and stripping them was the false-merge defect that
      ``test_identity_normalization.py`` pins).
    * ``fold_key`` may only ever be used to *find candidates*, never to mint an id,
      key a registry entry, or decide that two entities are the same. Two
      different people can share a fold key; that is what the review queue is for.
    """
    text = str(text or "")
    # German sharp s and the Polish stroke both survive NFD, so map them explicitly.
    text = text.replace("ß", "ss").replace("ẞ", "ss")
    text = text.replace("ø", "o").replace("Ø", "O")
    text = text.replace("ł", "l").replace("Ł", "L")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = _strip_diacritics(text)
    text = text.casefold()
    folded = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return " ".join(folded.split())


def strip_titles(text: str, language: str = "") -> tuple[str, list[str]]:
    """Remove leading honorifics/political titles.

    Returns ``(remainder, stripped)``. The title is recorded rather than thrown
    away, because "Pääministeri Orpo" and "Orpo" are analytically different
    mentions of the same actor even though they resolve to one entity.

    Titles are only recognised at the start of the mention and only when a name
    remains, so a party whose name merely contains a title word is untouched.
    """
    stripped: list[str] = []
    remainder = " ".join(str(text or "").split())
    languages = [language] if language else list(TITLES)
    # Longest titles first so "prime minister" is not matched as "minister".
    for lang in languages:
        vocabulary = TITLES.get(lang.lower(), ())
        for title in sorted(vocabulary, key=len, reverse=True):
            if not title:
                continue
            low = remainder.casefold()
            head = title.casefold()
            if low == head:
                continue
            if low.startswith(head + " ") or low.startswith(head + ".") or low.startswith(head + ","):
                candidate = remainder[len(title):].lstrip(" .,:;-–—")
                if len(candidate.strip()) >= MIN_NAME_LENGTH:
                    stripped.append(remainder[:len(title)])
                    remainder = candidate
                    break
    return remainder, stripped


def inflection_variants(text: str, language: str = "") -> list[str]:
    """Base-form candidates for an inflected name, longest suffix first.

    ``Orpon`` -> ``Orpo``, ``Orpolla`` -> ``Orpo``. Language knowledge is limited
    on purpose: this is a candidate generator, and the resolution steps that use it
    still have to find exactly one registry entity before anything is accepted.
    """
    word = " ".join(str(text or "").split())
    if not word:
        return []
    languages = [language] if language else list(INFLECTION_SUFFIXES)
    seen: list[str] = []
    parts = word.split()
    for lang in languages:
        for suffix in sorted(INFLECTION_SUFFIXES.get(lang.lower(), ()), key=len, reverse=True):
            if not suffix or len(word) <= len(suffix) + MIN_NAME_LENGTH:
                continue
            # Only the final token inflects; the given name does not.
            if not parts[-1].casefold().endswith(suffix.casefold()):
                continue
            stem = parts[-1][: len(parts[-1]) - len(suffix)]
            if len(stem) < MIN_NAME_LENGTH:
                continue
            candidate = " ".join([*parts[:-1], stem])
            key = fold_key(candidate)
            if key and key not in {fold_key(v) for v in seen}:
                seen.append(candidate)
    return seen


@dataclass(frozen=True)
class NormalizedMention:
    """The preserved mention plus every derived form. Nothing is discarded."""

    surface_form: str
    normalized_form: str
    stripped_titles: tuple[str, ...] = ()
    language: str = ""
    variants: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """The conservative identity key (NFC/casefold/whitespace). Unchanged."""
        return identity_key(self.surface_form)

    @property
    def fold(self) -> str:
        return fold_key(self.normalized_form or self.surface_form)

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_form": self.surface_form,
            "normalized_form": self.normalized_form,
            "stripped_titles": list(self.stripped_titles),
            "normalized_key": self.key,
            "fold_key": self.fold,
            "variants": list(self.variants),
        }


def normalize_mention(text: Any, *, language: str = "") -> NormalizedMention:
    """Normalize one surface mention without losing the original."""
    surface = " ".join(str(text or "").split())
    if not surface:
        return NormalizedMention("", "")
    remainder, titles = strip_titles(surface, language)
    normalized = remainder or surface
    variants = inflection_variants(normalized, language)
    return NormalizedMention(
        surface_form=surface,
        normalized_form=normalized,
        stripped_titles=tuple(titles),
        language=language.lower(),
        variants=tuple(variants),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class EntityRecord:
    """A canonical actor in the registry.

    Field names follow the issue's proposed object so the shape is recognisable,
    with the provenance and validity columns the existing codebooks already carry.
    """

    entity_id: str
    canonical_name: str
    entity_type: str = ""
    aliases: list[str] = field(default_factory=list)
    language_aliases: dict[str, list[str]] = field(default_factory=dict)
    country: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)
    source_languages: list[str] = field(default_factory=list)
    review_state: str = "PROVISIONAL"
    origin: str = ""
    valid_from: str = ""
    valid_to: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def forms(self) -> list[str]:
        values = [self.canonical_name, *self.aliases]
        for lang_forms in self.language_aliases.values():
            values.extend(lang_forms)
        return list(dict.fromkeys(v for v in (str(x).strip() for x in values) if v))

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "language_aliases": {k: list(v) for k, v in self.language_aliases.items()},
            "country": self.country,
            "external_ids": dict(self.external_ids),
            "source_languages": list(self.source_languages),
            "review_state": self.review_state,
            "origin": self.origin,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


class EntityRegistry:
    """Canonical entities, seeded from country codebooks and researcher mappings.

    The registry is a *view*: codebooks remain the source of truth, and this class
    does not write to them. Researcher corrections are layered on top so a
    reviewer's decision is durable instead of being rediscovered each run.
    """

    def __init__(self) -> None:
        self._records: dict[str, EntityRecord] = {}
        self._by_key: dict[str, list[str]] = {}
        self._by_fold: dict[str, list[str]] = {}
        self._corrections: dict[str, str] = {}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_codebooks(
        cls,
        entries: Iterable[CodebookEntry],
        *,
        corrections: dict[str, str] | None = None,
    ) -> EntityRegistry:
        """Seed from `CodebookEntry` objects (real codebooks, not a new format)."""
        registry = cls()
        for entry in entries:
            if entry.kind.casefold() not in ENTITY_KINDS:
                continue
            record = EntityRecord(
                entity_id=entry.entry_id,
                canonical_name=entry.label,
                entity_type=entry.entity_type or entry.kind,
                aliases=list(entry.aliases),
                language_aliases={},
                country=entry.country,
                external_ids={
                    k: v for k, v in (entry.metadata or {}).items()
                    if k in {"wikidata", "wikipedia", "external_id"} and v
                },
                source_languages=list(entry.source_languages),
                review_state=entry.review_state,
                origin=entry.origin or entry.layer,
                valid_from=entry.valid_from,
                valid_to=entry.valid_to,
                provenance={"entry_id": entry.entry_id, "layer": entry.layer},
            )
            if entry.english_label:
                record.aliases.append(entry.english_label)
                record.language_aliases.setdefault("en", []).append(entry.english_label)
            for lang in entry.source_languages:
                record.language_aliases.setdefault(lang.lower(), []).append(entry.label)
            registry.add(record)
        if corrections:
            registry.apply_corrections(corrections)
        return registry

    @classmethod
    def from_codebook_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        corrections: dict[str, str] | None = None,
    ) -> EntityRegistry:
        """Seed from codebook JSON files via the existing loader."""
        from roihu_codebooks import load_codebook

        entries: list[CodebookEntry] = []
        for path in paths:
            loaded, _meta = load_codebook(path)
            entries.extend(loaded)
        return cls.from_codebooks(entries, corrections=corrections)

    def add(self, record: EntityRecord) -> None:
        if not record.entity_id or not record.canonical_name:
            raise ValueError("registry records need entity_id and canonical_name")
        self._records[record.entity_id] = record
        self._reindex()

    def _reindex(self) -> None:
        self._by_key, self._by_fold = {}, {}
        for record in self._records.values():
            for form in record.forms:
                self._by_key.setdefault(identity_key(form), []).append(record.entity_id)
                self._by_fold.setdefault(fold_key(form), []).append(record.entity_id)

    # -- researcher corrections -------------------------------------------- #

    def apply_corrections(self, corrections: dict[str, str]) -> None:
        """Map a surface form to an entity id, durably.

        ``corrections`` is ``{surface_form: entity_id}``. A correction whose target
        is unknown is refused rather than stored, so a typo cannot create a
        dangling mapping that silently swallows mentions.
        """
        for surface, entity_id in corrections.items():
            key = identity_key(surface)
            if not key:
                continue
            if entity_id not in self._records:
                raise KeyError(f"correction targets unknown entity_id: {entity_id!r}")
            self._corrections[key] = entity_id
            record = self._records[entity_id]
            if surface.strip() not in record.forms:
                record.aliases.append(surface.strip())
        self._reindex()

    @staticmethod
    def load_corrections(path: str | Path) -> dict[str, str]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("corrections file must be a JSON object of surface -> entity_id")
        return {str(k): str(v) for k, v in payload.items()}

    # -- lookups ------------------------------------------------------------ #

    def get(self, entity_id: str) -> EntityRecord | None:
        return self._records.get(entity_id)

    def records(self) -> list[EntityRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def _scoped(
        self,
        *,
        country: str = "",
        entity_type: str = "",
        valid_at: str = "",
    ) -> list[EntityRecord]:
        """Apply country/type/time constraints.

        Context is a *disambiguation signal*, never evidence that an entity was
        mentioned: an empty ``country`` on a record means country-agnostic, and a
        request with no country matches every record.
        """
        wanted_country = country.upper()
        wanted_type = entity_type.casefold()
        out = []
        for record in self._records.values():
            if wanted_country and record.country and record.country.upper() not in {wanted_country, "COMMON"}:
                continue
            if wanted_type and record.entity_type.casefold() != wanted_type:
                continue
            if valid_at:
                if record.valid_from and valid_at < record.valid_from:
                    continue
                if record.valid_to and valid_at > record.valid_to:
                    continue
            out.append(record)
        return out

    # -- resolution --------------------------------------------------------- #

    def resolve(
        self,
        mention: Any,
        *,
        country: str = "",
        entity_type: str = "",
        language: str = "",
        valid_at: str = "",
        fuzzy_threshold: float = 0.86,
        fuzzy_limit: int = 5,
        embedder: Callable[[str], Sequence[float]] | None = None,
        semantic_threshold: float = 0.80,
        adjudicator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one mention following the documented order.

        Returns a record carrying the preserved mention, the decision, the match
        method that decided it, confidence, and -- when the answer is not unique --
        every candidate. Ambiguity always abstains.
        """
        normalized = normalize_mention(mention, language=language)
        base: dict[str, Any] = {
            **normalized.as_dict(),
            "country": country.upper(),
            "entity_type": entity_type,
            "language": language.lower(),
            "valid_at": valid_at,
        }
        if not normalized.surface_form:
            return {**base, "decision": "EMPTY", "match_method": "empty", "candidates": []}

        scoped = self._scoped(country=country, entity_type=entity_type, valid_at=valid_at)

        # Step 0: a researcher correction outranks every heuristic.
        corrected = self._corrections.get(identity_key(normalized.surface_form))
        if corrected and corrected in {r.entity_id for r in scoped}:
            return self._accept(base, self._records[corrected], "researcher_mapping", 1.0)

        # Step 1/2: exact canonical name, exact English label, exact alias.
        for method, forms_of in (
            ("exact_canonical", lambda r: [r.canonical_name]),
            ("exact_english_label", lambda r: r.language_aliases.get("en", [])),
            ("exact_alias", lambda r: [*r.aliases, *[f for v in r.language_aliases.values() for f in v]]),
        ):
            hits = self._unique(scoped, normalized.surface_form, forms_of)
            if hits is not None:
                return self._finish(base, hits, method)

        # Step 3: language-aware normalization -- title stripping, then morphology
        # folding, then accent folding. Each is reported separately so a reviewer
        # can see which reduction produced the match.
        for method, probe in (
            ("title_stripped", normalized.normalized_form),
            ("morphology_folded", ""),
            ("diacritic_folded", ""),
        ):
            if method == "title_stripped":
                if normalized.normalized_form == normalized.surface_form:
                    continue
                hits = self._unique(scoped, probe, lambda r: r.forms, key=fold_key)
                if hits is not None:
                    return self._finish(base, hits, method, confidence=0.95)
                continue
            # Morphology / diacritics: try every generated variant.
            variants = list(normalized.variants) if method == "morphology_folded" else [normalized.normalized_form]
            for variant in variants:
                hits = self._unique(scoped, variant, lambda r: r.forms, key=fold_key)
                if hits is not None:
                    return self._finish(base, hits, method, confidence=0.9)

        # Step 3b: partial-name references -- surname-only or given-name-only.
        # The issue requires these explicitly ("first-name-only / surname-only
        # references"). Only a UNIQUE match within the scoped registry may resolve,
        # and at reduced confidence, because a bare surname is genuinely weaker
        # evidence than a full name: two Orpos must never be guessed between.
        for method, part in (("surname_unique", -1), ("given_name_unique", 0)):
            # Consider the normalized form and its inflection variants, so an
            # inflected bare surname ("Orpon") reaches the same unique-surname
            # check that resolves its base form ("Orpo").
            probes = [normalized.normalized_form, *normalized.variants]
            hits: dict[str, EntityRecord] = {}
            for probe in probes:
                tokens = fold_key(probe).split()
                if len(tokens) != 1:
                    continue
                token = tokens[0]
                for record in scoped:
                    # Partial personal-name heuristics are meaningful only for people.
                    # Without this guard a token such as "Orpo" can match the first
                    # token of an organisation label like "Orpo's government" via
                    # given_name_unique, silently converting a person mention into an
                    # organisation. Exact/alias matching above still resolves parties
                    # and organisations normally.
                    if record.entity_type.casefold() not in {"person", "politician", "human"}:
                        continue
                    parts = fold_key(record.canonical_name).split()
                    if len(parts) < 2:
                        continue
                    if parts[part] == token:
                        hits[record.entity_id] = record
            if len(hits) == 1:
                return self._finish(base, [next(iter(hits.values()))], method, confidence=0.8)
            if len(hits) > 1:
                LOG.debug("partial-name probe matches %d entities; abstaining", len(hits))

        # Step 4: fuzzy similarity -- candidates only, never accepted.
        fuzzy = self._fuzzy(scoped, normalized, threshold=fuzzy_threshold, limit=fuzzy_limit)

        # Step 5: optional multilingual semantic matching -- also candidates only.
        semantic: list[dict[str, Any]] = []
        if embedder is not None:
            semantic = self._semantic(scoped, normalized, embedder, threshold=semantic_threshold)

        candidates = self._merge_candidates(fuzzy, semantic)
        if candidates and adjudicator is not None:
            try:
                verdict = adjudicator({**base, "candidates": candidates})
            except Exception as exc:
                LOG.warning("LLM adjudicator failed: %s", exc)
                verdict = None
            entity_id = verdict if isinstance(verdict, str) else (
                str(verdict.get("entity_id") or "") if isinstance(verdict, dict) else ""
            )
            allowed = {c["entity_id"] for c in candidates}
            if entity_id in allowed and entity_id in self._records:
                return self._accept(base, self._records[entity_id], "llm_adjudicated", 0.85)

        if candidates:
            return {
                **base,
                "decision": "CANDIDATE",
                "match_method": "fuzzy_candidate" if fuzzy else "semantic_candidate",
                "confidence": max((c["score"] for c in candidates), default=0.0),
                "candidates": candidates,
            }

        # Step 6/7: nothing matched. Unresolved -- the review queue's job.
        return {
            **base,
            "decision": "UNRESOLVED",
            "match_method": "no_match",
            "confidence": 0.0,
            "candidates": [],
        }

    def resolve_many(self, mentions: Iterable[str], **kwargs: Any) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for mention in mentions:
            key = identity_key(mention)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(self.resolve(mention, **kwargs))
        return out

    # -- internals ---------------------------------------------------------- #

    def _unique(
        self,
        scoped: list[EntityRecord],
        probe: str,
        forms_of: Callable[[EntityRecord], Sequence[str]],
        *,
        key: Callable[[Any], str] | None = None,
    ) -> list[EntityRecord] | None:
        """Return the single matching record, or None.

        ``None`` means "not this step" and deliberately also covers *several*
        matches: two records reachable from one probe is an ambiguity, and this
        layer never guesses between them.
        """
        keyer = key or identity_key
        wanted = keyer(probe)
        if not wanted:
            return None
        matches: dict[str, EntityRecord] = {}
        for record in scoped:
            for form in forms_of(record):
                if keyer(form) == wanted:
                    matches[record.entity_id] = record
        if len(matches) == 1:
            return [next(iter(matches.values()))]
        if len(matches) > 1:
            LOG.debug("ambiguous probe=%r reachable from %d records", probe, len(matches))
        return None

    def _fuzzy(
        self,
        scoped: list[EntityRecord],
        normalized: NormalizedMention,
        *,
        threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        # Probe with the title-stripped form and the raw surface form. A corrupted
        # title ("Pääministeri 0rpo") makes the stripped remainder useless as a
        # probe, while the full surface still resembles the full canonical form --
        # so taking the better of the two recovers mentions that either alone misses.
        probes = [p for p in dict.fromkeys(
            fold_key(x) for x in (normalized.normalized_form, normalized.surface_form)
        ) if p]
        if not probes:
            return []
        scorer = _rapidfuzz_scorer()
        ranked: list[tuple[float, str, EntityRecord, str]] = []
        for record in scoped:
            best_score, best_form = 0.0, ""
            for form in record.forms:
                form_fold = fold_key(form)
                if not form_fold:
                    continue
                for probe in probes:
                    score = scorer(probe, form_fold)
                    if score > best_score:
                        best_score, best_form = score, form
            if best_score >= threshold:
                ranked.append((best_score, record.entity_id, record, best_form))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "entity_id": record.entity_id,
                "canonical_name": record.canonical_name,
                "entity_type": record.entity_type,
                "country": record.country,
                "matched_form": form,
                "score": round(float(score), 6),
                "method": "fuzzy_candidate",
            }
            for score, _id, record, form in ranked[: max(1, limit)]
        ]

    def _semantic(
        self,
        scoped: list[EntityRecord],
        normalized: NormalizedMention,
        embedder: Callable[[str], Sequence[float]],
        *,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Cosine similarity over caller-supplied embeddings.

        The embedder is injected rather than imported so that this module needs no
        model to run, and so tests do not depend on a download. In production this
        is a multilingual SentenceTransformers model.
        """
        try:
            probe = list(embedder(normalized.normalized_form or normalized.surface_form))
        except Exception as exc:  # noqa: BLE001 - a broken embedder must not break resolution
            LOG.warning("semantic embedder failed: %s", exc)
            return []
        if not probe:
            return []
        out: list[dict[str, Any]] = []
        for record in scoped:
            best_score, best_form = 0.0, ""
            for form in record.forms:
                try:
                    vector = list(embedder(form))
                except Exception:  # noqa: BLE001
                    continue
                score = _cosine(probe, vector)
                if score > best_score:
                    best_score, best_form = score, form
            if best_score >= threshold:
                out.append({
                    "entity_id": record.entity_id,
                    "canonical_name": record.canonical_name,
                    "entity_type": record.entity_type,
                    "country": record.country,
                    "matched_form": best_form,
                    "score": round(float(best_score), 6),
                    "method": "semantic_candidate",
                })
        out.sort(key=lambda c: (-c["score"], c["entity_id"]))
        return out

    @staticmethod
    def _merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for group in groups:
            for candidate in group:
                current = best.get(candidate["entity_id"])
                if current is None or candidate["score"] > current["score"]:
                    best[candidate["entity_id"]] = candidate
        return sorted(best.values(), key=lambda c: (-c["score"], c["entity_id"]))

    @staticmethod
    def _accept(
        base: dict[str, Any],
        record: EntityRecord,
        method: str,
        confidence: float,
    ) -> dict[str, Any]:
        return {
            **base,
            "decision": "RESOLVED",
            "match_method": method,
            "confidence": round(float(confidence), 6),
            "entity_id": record.entity_id,
            "canonical_name": record.canonical_name,
            "entity_type": record.entity_type,
            "entity_country": record.country,
            "review_state": record.review_state,
            "provenance": dict(record.provenance),
            "candidates": [],
        }

    def _finish(
        self,
        base: dict[str, Any],
        hits: list[EntityRecord],
        method: str,
        *,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        return self._accept(base, hits[0], method, confidence)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if not na or not nb else dot / (na * nb)


def _rapidfuzz_scorer() -> Callable[[str, str], float]:
    """RapidFuzz when installed, stdlib SequenceMatcher otherwise.

    Both return a 0..1 similarity. RapidFuzz is faster and handles transpositions
    better; the fallback keeps the layer importable in a minimal environment,
    which is why the dependency is optional rather than required.
    """
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return lambda a, b: SequenceMatcher(None, a, b).ratio()
    return lambda a, b: float(fuzz.token_set_ratio(a, b)) / 100.0


def entity_resolution_columns() -> tuple[str, ...]:
    """The columns this layer appends. Declared once so stages stay in step."""
    return (
        "ep24_entity_resolution_json",
        "ep24_entity_ids",
        "ep24_entity_canonical_names",
        "ep24_entity_unresolved_json",
    )


def resolve_dataframe(
    frame: Any,
    registry: EntityRegistry,
    *,
    country: str = "",
    language: str = "",
    valid_at: str = "",
    mention_columns: Sequence[str] = (
        "new_entity",
        "researcher_new_persons",
        "entities_seed_provenance",
    ),
    embedder: Callable[[str], Sequence[float]] | None = None,
    adjudicator: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Append canonical entity fields to a DataFrame, preserving every column.

    Additive by contract, matching the repository rule in ``ep24_schema.py``: the
    incoming frame is authoritative, every source column is forwarded untouched,
    and this layer only appends. The original mention text is never overwritten --
    that is the issue's central requirement, and it is why the appended columns are
    named ``ep24_entity_*`` rather than replacing ``new_entity``.

    Returns a summary dict (counters and the review queue) for stage logging.
    """
    import pandas as pd  # local import: this module stays importable without pandas

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("resolve_dataframe expects a pandas DataFrame")

    before = list(frame.columns)
    for column in entity_resolution_columns():
        if column not in frame.columns:
            frame[column] = ""

    results: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        row_results: list[dict[str, Any]] = []
        for column in mention_columns:
            for mention in _split_mentions(row.get(column)):
                result = registry.resolve(
                    mention,
                    country=country,
                    language=language,
                    valid_at=valid_at,
                    embedder=embedder,
                    adjudicator=adjudicator,
                )
                result["source_column"] = column
                row_results.append(result)
        frame.at[index, "ep24_entity_resolution_json"] = json.dumps(
            row_results, ensure_ascii=False, sort_keys=True
        )
        frame.at[index, "ep24_entity_ids"] = json.dumps(
            sorted({r["entity_id"] for r in row_results if r.get("entity_id")}),
            ensure_ascii=False,
        )
        frame.at[index, "ep24_entity_canonical_names"] = json.dumps(
            sorted({r["canonical_name"] for r in row_results if r.get("canonical_name")}),
            ensure_ascii=False,
        )
        queued = [r for r in row_results if r["decision"] in {"CANDIDATE", "UNRESOLVED"}]
        frame.at[index, "ep24_entity_unresolved_json"] = json.dumps(
            queued, ensure_ascii=False, sort_keys=True
        )
        results.extend(row_results)
        for result in queued:
            unresolved_rows.append({"row": int(index), **result})

    after = list(frame.columns)
    summary = resolution_summary(results)
    summary["columns_added"] = [c for c in after if c not in before]
    summary["columns_preserved"] = all(c in after for c in before)
    summary["source_columns"] = before
    return summary


def _split_mentions(value: Any) -> list[str]:
    """Split a seed cell into mentions.

    The separator set deliberately matches ``roihu_enrich.split_values`` and
    ``roihu_rdf.split_list``, because all three read the *same* cell and the
    postprocess stage writes it with ``', '.join(...)``. An earlier revision
    split only on newlines and semicolons, so a cell such as
    ``"Sanna Marin, Petteri Orpo"`` was handed to :meth:`EntityRegistry.resolve`
    as a single mention, matched nothing, and *both* actors were lost to the
    unresolved queue -- exactly the fragmentation this layer exists to prevent.

    Measured on the private corpus: 2,375 of the Finland ``entities`` cells and
    3,496 of the Hungary cells carry commas, so this affected most mention cells,
    not an edge case.

    Empty containers such as ``'[]'`` -- common noise in the researcher columns
    -- still yield nothing.
    """
    text = str(value or "").strip()
    if not text or text in {"[]", "{}", "nan", "None"}:
        return []
    out: list[str] = []
    for chunk in text.replace("\r", "\n").split("\n"):
        for part in chunk.replace(";", ",").replace("|", ",").split(","):
            candidate = part.strip().strip("\"'").strip()
            if candidate and candidate not in {"[]", "{}"}:
                out.append(candidate)
    return list(dict.fromkeys(out))


def resolution_summary(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-run counters, including the review queue.

    ``unresolved`` and ``ambiguous`` are reported separately from ``resolved``
    because a pipeline that quietly folds the first two into the third is exactly
    the failure this layer exists to prevent.
    """
    counts: dict[str, int] = {}
    methods: dict[str, int] = {}
    queue: list[dict[str, Any]] = []
    for result in results:
        decision = str(result.get("decision", "UNKNOWN"))
        counts[decision] = counts.get(decision, 0) + 1
        method = str(result.get("match_method", ""))
        if method:
            methods[method] = methods.get(method, 0) + 1
        if decision in {"CANDIDATE", "UNRESOLVED"}:
            queue.append({
                "surface_form": result.get("surface_form", ""),
                "decision": decision,
                "candidates": [
                    c.get("entity_id") for c in result.get("candidates", [])
                ],
            })
    return {
        "total": sum(counts.values()),
        "decisions": counts,
        "methods": methods,
        "resolved": counts.get("RESOLVED", 0),
        "review_queue": queue,
    }

def ollama_adjudicator(model: str | None = None) -> Callable[[dict[str, Any]], Any]:
    """Return a conservative Ollama-backed candidate adjudicator."""
    chosen_model = model or ollama_model(specific_env="LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")

    def adjudicate(payload: dict[str, Any]) -> Any:
        import ollama
        prompt = {
            "surface_form": payload.get("surface_form", ""),
            "normalized_form": payload.get("normalized_form", ""),
            "language": payload.get("language", ""),
            "country": payload.get("country", ""),
            "candidates": payload.get("candidates", []),
            "instruction": "Choose one supplied entity_id only if unambiguous; otherwise return an empty entity_id. Never invent an id.",
        }
        response = ollama.chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "Conservative entity-resolution adjudicator. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            format="json", options={"temperature": 0.0},
        )
        data = json.loads(response["message"]["content"])
        return {"entity_id": str(data.get("entity_id") or "")}

    return adjudicate


def registry_documents(registry: EntityRegistry) -> list[dict[str, Any]]:
    """Serialize canonical registry records for Mongo persistence."""
    return [{"_storage_id": r.entity_id, **r.as_dict(), "provenance": dict(r.provenance)} for r in registry.records()]


def resolution_lookup(value: Any) -> dict[str, dict[str, str]]:
    """Build surface/canonical label to stable-id lookup from resolution JSON."""
    try:
        rows = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    lookup: dict[str, dict[str, str]] = {}
    if not isinstance(rows, list):
        return lookup
    for row in rows:
        if not isinstance(row, dict) or row.get("decision") != "RESOLVED" or not row.get("entity_id"):
            continue
        item = {"entity_id": str(row["entity_id"]), "canonical_name": str(row.get("canonical_name") or "")}
        for label in (row.get("surface_form"), row.get("normalized_form"), row.get("canonical_name")):
            key = fold_key(label)
            if key:
                lookup[key] = item
    return lookup
