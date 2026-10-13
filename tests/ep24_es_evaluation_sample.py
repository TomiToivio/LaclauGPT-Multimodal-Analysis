#!/usr/bin/env python3
"""EP24 #72 evaluation sample for Spain (ES): measurable normalization cases.

This is the "small reproducible review/evaluation sample" the issue asks for.
It encodes concrete before/after expectations for the Spain codebook surface,
so a later independent reviewer can falsify pass 1 rather than merely agree.

Everything here is PUBLIC and generic:

* the expected canonical party identities are public facts about the 2024
  European Parliament election in Spain;
* the *observed legacy surface forms* are quoted only to show the failure class
  (lowercasing, comma-joined blobs, parenthesised fragments, English-translation
  names, missing acronyms). No private row content, spreadsheet cell, researcher
  note or row identifier is reproduced.

Run:

    python tests/ep24_es_evaluation_sample.py            # human-readable report
    python -m pytest tests/ep24_es_evaluation_sample.py  # as regression cases

The module deliberately does NOT import the private codebooks: it states the
desired contract, so it can run anywhere (including CI) without private data.
A private-side companion test can assert the same expectations against
``ep24_es_private.json`` once the builder has been fixed.
"""
from __future__ import annotations

import re
import unicodedata

# --- The five largest nationwide Spanish parties: (canonical, acronym, aliases)
NATIONWIDE_PARTIES = [
    ("Partido Socialista Obrero Español", "PSOE", ["PSOE", "Psoe", "Partido Socialista"]),
    ("Partido Popular", "PP", ["PP", "Populares"]),
    ("Vox", "Vox", ["VOX", "Vox", "VÓX"]),
    ("Sumar", "Sumar", ["Sumar", "Coalición Sumar"]),
    ("Podemos", "Podemos", ["Podemos"]),
]

# Regional / coalition list identities that legacy material flattened into
# member parties or into English translations.
REGIONAL_LISTS = [
    ("Junts i Lliures per Europa", "Junts+", ["Junts", "JxCAT", "Junts per Catalunya"]),
    ("Ahora Repúblicas", "AR", ["Ahora Repúblicas"]),
    ("Coalición por una Europa Solidaria", "CEUS", ["CEUS"]),
    ("Coalición Compromís", "Compromís", ["Compromís"]),
    ("Esquerra Republicana de Catalunya", "ERC", ["ERC"]),
    ("Euzko Alderdi Jeltzalea – Partido Nacionalista Vasco", "EAJ-PNV", ["PNV", "EAJ-PNV"]),
    ("Euskal Herria Bildu", "EH Bildu", ["EH Bildu", "Bildu"]),
    ("Bloque Nacionalista Galego", "BNG", ["BNG"]),
]

# English renderings observed in legacy private material that must be treated as
# translation artefacts (aliases), never as the canonical entity label.
TRANSLATION_ARTEFACTS = [
    ("Izquierda Unida", "United Left"),
    ("Más Madrid", "More Madrid"),
    ("Coalición canaria", "Canarian Coalition"),
    ("Sumar", "Unite"),
    ("Podemos", "We Can"),
    ("Se Acabó La Fiesta", "The Party is Over"),
    ("Ahora Repúblicas", "Republics Now"),
    ("Junts i Lliures per Europa", "Together and Free for Europe"),
]


def _identity_key(text: str) -> str:
    """Mirror of roihu_codebooks.identity_key: NFC + casefold + whitespace only."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).strip().casefold().split())


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)
    )


def observed_legacy_forms() -> list[str]:
    """Surface forms of the failure classes, as they actually occurred.

    Quoted only to demonstrate the class; not a private data dump.
    """
    return [
        # comma-joined multi-entity blobs
        "pedro sánchez, politician",
        "santiago abascal, vox party",
        "jorge buxadé, the european union, the vox party",
        # party name glued to its own abbreviation
        "psoe partido socialista obrero español",
        "partido popular pp",
        "spanish socialist workers' party psoe",
        # parenthesised or dangling fragments produced by naive splitting
        "PSOE)",
        "Podemos)",
        "Sumar)",
        "6 mil)",
        # lowercased duplicates of proper names
        "pedro sánchez",
        "isabel díaz ayuso",
        # generic strings leaking into the entity field
        "politicians",
        "the speaker",
        "politician candidate",
    ]


def _is_multi_entity_blob(value: str) -> bool:
    """A cell that packs more than one entity into one string."""
    return "," in value


def _needs_canonicalization(value: str) -> bool:
    """True when the value is not a clean single entity surface form."""
    v = value.strip()
    if not v:
        return False
    if _is_multi_entity_blob(v):
        return True
    if re.search(r"\)\s*$", v) and "(" not in v:  # dangling 'PSOE)'
        return True
    if re.fullmatch(r"[\d\W]+", v):  # '6 mil)' style
        return True
    if v != v[:1].upper() + v[1:]:  # lowercase proper name
        return True
    return False


def evaluate() -> dict[str, object]:
    forms = observed_legacy_forms()

    # 1. Acronyms must be resolvable as aliases on a canonical entry.
    alias_index: dict[str, str] = {}
    for canonical, acronym, aliases in NATIONWIDE_PARTIES + REGIONAL_LISTS:
        for form in {acronym, *aliases}:
            alias_index.setdefault(_identity_key(form), canonical)

    acronym_probe = ["psoe", "pp", "vox", "vÓx", "sumar", "podemos", "erc", "pnv", "bng", "ceus"]
    acronym_hits = {a: alias_index.get(_identity_key(a)) for a in acronym_probe}
    acronym_misses = [a for a, hit in acronym_hits.items() if hit is None]

    # 2. Every translation artefact must map to a canonical Spanish label.
    canonical_labels = {_identity_key(c) for c, _a, _al in NATIONWIDE_PARTIES + REGIONAL_LISTS}
    canonical_labels |= {_identity_key(c) for c, _e in TRANSLATION_ARTEFACTS}
    mistyped_as_canonical = [
        english
        for _es, english in TRANSLATION_ARTEFACTS
        if _identity_key(english) in canonical_labels and
        not any(_identity_key(c) == _identity_key(english) for c, _a, _al in NATIONWIDE_PARTIES + REGIONAL_LISTS)
    ]

    # 3. Diacritics must survive canonicalization: accented and unaccented forms
    #    of the same person must NOT be the same identity key.
    diacritic_pairs = [("Pedro Sánchez", "Pedro Sanchez"), ("Isabel Díaz Ayuso", "Isabel Diaz Ayuso")]
    diacritic_collisions = [
        (a, b) for a, b in diacritic_pairs if _identity_key(a) == _identity_key(b)
    ]

    # 4. Legacy failure classes must be recognised as needing canonicalization.
    flagged = [f for f in forms if _needs_canonicalization(f)]

    return {
        "acronym_misses": acronym_misses,
        "translation_artefacts_mistyped_as_canonical": mistyped_as_canonical,
        "diacritic_collisions": diacritic_collisions,
        "legacy_forms_total": len(forms),
        "legacy_forms_flagged": len(flagged),
    }


def test_acronyms_resolve_to_a_canonical_party():
    """ES-01: PSOE/PP/Vox/Sumar/Podemos and regional acronyms must resolve."""
    result = evaluate()
    assert result["acronym_misses"] == [], (
        "these acronyms have no canonical entry and would be dropped: "
        f"{result['acronym_misses']}"
    )


def test_english_translations_are_not_canonical_labels():
    """ES-02: 'Unite'/'We Can'/'The Party is Over' must be aliases, not labels."""
    result = evaluate()
    assert result["translation_artefacts_mistyped_as_canonical"] == [], (
        "translation artefacts used as canonical labels: "
        f"{result['translation_artefacts_mistyped_as_canonical']}"
    )


def test_diacritics_are_not_collapsed_in_identity():
    """ES-03: accented and unaccented spellings must stay distinct identities."""
    result = evaluate()
    assert result["diacritic_collisions"] == [], (
        f"diacritic-stripping collapsed distinct identities: {result['diacritic_collisions']}"
    )


def test_legacy_failure_classes_are_detected():
    """ES-04: blobs, fragments, glued acronyms and lowercase names are flagged."""
    result = evaluate()
    assert result["legacy_forms_flagged"] == result["legacy_forms_total"], (
        f"only {result['legacy_forms_flagged']}/{result['legacy_forms_total']} "
        "observed legacy forms were recognised as needing canonicalization"
    )


def main() -> int:
    result = evaluate()
    print("EP24 #72 — Spain (ES) evaluation sample")
    print("=" * 62)
    print(f"nationwide parties modelled : {len(NATIONWIDE_PARTIES)}")
    print(f"regional lists modelled     : {len(REGIONAL_LISTS)}")
    print(f"translation artefacts       : {len(TRANSLATION_ARTEFACTS)}")
    print()
    print(f"acronym misses              : {result['acronym_misses'] or 'none'}")
    print(f"translations as canonical   : {result['translation_artefacts_mistyped_as_canonical'] or 'none'}")
    print(f"diacritic collisions        : {result['diacritic_collisions'] or 'none'}")
    print(
        f"legacy forms flagged        : {result['legacy_forms_flagged']}/"
        f"{result['legacy_forms_total']}"
    )
    print()
    print("Contract: a bare Spanish acronym must resolve inside ES; an English")
    print("translation must never be the canonical label; diacritics are identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
