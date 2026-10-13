"""Known-issue #1 (§1): reconcile memory vs codebook label normalization.

The two modules must agree on *identity* while both preserving the distinctions
that matter for discourse analysis. The rule pinned here:

* the identity key is conservative — Unicode NFC + casefold + whitespace
  collapse — so a concept written with stray or non-breaking whitespace cannot
  produce two identities;
* accents, parenthesized qualifiers and punctuation are NOT stripped, because
  erasing them was the false-merge defect (known-issue #2) that PR #17 made
  visible;
* an already-recorded or explicitly supplied entry id is never recomputed, so
  existing data cannot be silently re-pointed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402
import roihu_memory as rm  # noqa: E402


def _old_entry_id(label: str, kind: str = "entity", country: str = "FI", disamb: str = "") -> str:
    """The pre-fix seed, to prove clean labels do not move."""
    seed = "|".join((country.upper(), kind, label.casefold(), disamb.casefold()))
    return "CB-" + hashlib.sha256(seed.encode()).hexdigest()[:18]


def _new_entry_id(label: str, kind: str = "entity", country: str = "FI", disamb: str = "") -> str:
    seed = "|".join((country.upper(), kind, cb.identity_key(label), cb.identity_key(disamb)))
    return "CB-" + hashlib.sha256(seed.encode()).hexdigest()[:18]


def _write(root: Path, name: str, entries: list[dict]) -> None:
    (root / "codebooks").mkdir(parents=True, exist_ok=True)
    (root / "codebooks" / name).write_text(
        json.dumps({"schema": "fixture", "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# the two modules agree on identity
# --------------------------------------------------------------------------- #

def test_identity_key_matches_memory_surface_key():
    """One concept must not have two identities across the two modules."""
    for label in ("Election", "Election ", "  Election", "Election\u00a0",
                  "Grüne", "Łukasz Köhler", "Åsa (MEP)", "Straße", "STRASSE",
                  "Puolue", "Petteri Orpo"):
        assert cb.identity_key(label) == rm.surface_key(label), label


def test_whitespace_and_unicode_form_variants_share_an_id():
    """These differ only by whitespace, so they are one concept."""
    for a, b in (("Election", "Election "),
                 ("Election", "  Election"),
                 ("Election", "Election\u00a0"),
                 ("Puolue", "Puolue\t"),
                 ("Petteri Orpo", "Petteri  Orpo")):
        assert cb.identity_key(a) == cb.identity_key(b), (a, b)


def test_nfc_normalisation_is_applied():
    """A decomposed form must hash the same as its composed form."""
    composed = "Café"                      # U+00E9
    decomposed = "Cafe\u0301"              # e + combining acute
    assert composed != decomposed
    assert cb.identity_key(composed) == cb.identity_key(decomposed)


# --------------------------------------------------------------------------- #
# the deliberate distinctions are preserved
# --------------------------------------------------------------------------- #

def test_accents_and_qualifiers_are_not_stripped():
    """Erasing these was the false-merge defect; identity must keep them apart."""
    assert cb.identity_key("Grüne") != cb.identity_key("Grune")
    assert cb.identity_key("Åsa (MEP)") != cb.identity_key("Asa")
    assert cb.identity_key("Puolue (A)") != cb.identity_key("Puolue (B)")
    assert cb.identity_key("Sanchez") != cb.identity_key("Sánchez")


def test_original_label_is_never_normalised_in_the_entry():
    """The key is normalized; the stored label is verbatim."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_finland_private.json", [
        {"kind": "entity", "label": "Grüne ", "country": "FI", "definition": "d"},
    ])
    _write(root, "ep24_common_private.json", [])
    entries, _meta = cb.load_profile(root, "FI")
    assert len(entries) == 1
    # Trailing space is stripped by the loader's `_clean`, but the diacritic and
    # the case are preserved rather than casefolded into the stored label.
    assert entries[0].label == "Grüne"


# --------------------------------------------------------------------------- #
# no id movement for existing data
# --------------------------------------------------------------------------- #

def test_clean_labels_keep_their_existing_ids():
    """Ids already derived from clean labels must not change."""
    for label in ("Election", "Puolue", "Petteri Orpo", "Grüne",
                  "Łukasz Köhler", "Åsa (MEP)"):
        assert _new_entry_id(label) == _old_entry_id(label), label


def test_explicit_id_is_never_recomputed():
    """An entry that carries its own id keeps it, whatever the label looks like."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_finland_private.json", [
        {"kind": "entity", "id": "fi-human-1", "label": "Grüne ",
         "country": "FI", "definition": "d"},
    ])
    _write(root, "ep24_common_private.json", [])
    entries, _meta = cb.load_profile(root, "FI")
    assert [e.entry_id for e in entries] == ["fi-human-1"]


def test_only_whitespace_drift_moves_an_id():
    """The change is narrow: only ids whose label had stray whitespace differ."""
    assert _new_entry_id("Election") == _old_entry_id("Election")
    # This is the one class that legitimately changes, and it is the point.
    assert _new_entry_id("Election ") != _old_entry_id("Election ")


# --------------------------------------------------------------------------- #
# the merge now sees whitespace variants as one entry
# --------------------------------------------------------------------------- #

def test_merge_detects_whitespace_variant_as_a_duplicate():
    """Two same-country entries differing only by whitespace are one entry.

    Note the country scoping is load-bearing: a `common`-layer entry defaults to
    country `COMMON` and therefore keys separately from a `FI` entry. That is the
    intended country isolation, so the duplicate check below stays within one
    country layer.
    """
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    # Same country, same concept, one with a trailing NBSP.
    _write(root, "ep24_finland_private.json", [
        {"kind": "entity", "label": "Election", "country": "FI", "definition": "a"},
        {"kind": "entity", "label": "Election\u00a0", "country": "FI", "definition": "b"},
    ])
    entries, _meta = cb.load_profile(root, "FI")
    assert len(entries) == 1, "a whitespace variant must not create a second entry"


def test_merge_key_is_normalised_within_a_country():
    """The merge key collapses whitespace variants but keeps countries apart."""
    common = ("entity", cb.identity_key("Election"), "COMMON")
    fi_plain = ("entity", cb.identity_key("Election"), "FI")
    fi_nbsp = ("entity", cb.identity_key("Election\u00a0"), "FI")
    assert fi_plain == fi_nbsp, "whitespace variants must share a merge key"
    assert fi_plain != common, "country scoping must still separate them"


def test_country_still_scopes_identity():
    """Identical labels in different countries must stay distinct."""
    assert cb.identity_key("Election") == cb.identity_key("Election")
    # Country participates in the id seed, not the key.
    assert _new_entry_id("Election", country="FI") != _new_entry_id("Election", country="PL")


def test_kind_still_scopes_identity():
    assert _new_entry_id("Election", kind="entity") != _new_entry_id("Election", kind="topic")


# --------------------------------------------------------------------------- #
# the real private codebooks still load, unchanged
# --------------------------------------------------------------------------- #

def test_alias_map_uses_the_shared_key():
    """Alias collision detection must use the same key, or it misses variants.

    The reported key is `ambiguous_forms`. Two entries sharing an alias that
    differs only by whitespace must be flagged, which only happens if the alias
    map is keyed by the same normalization as identity.
    """
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_finland_private.json", [
        {"kind": "entity", "label": "Alpha", "aliases": ["Shared "], "country": "FI"},
        {"kind": "entity", "label": "Beta", "aliases": ["Shared"], "country": "FI"},
    ])
    _entries, meta = cb.load_profile(root, "FI")
    ambiguous = meta.get("ambiguous_forms") or []
    assert any("shared" in str(a).casefold() for a in ambiguous), meta


# --------------------------------------------------------------------------- #
# Croatia (#72): diacritics and country profile are deliberate constraints
# --------------------------------------------------------------------------- #

def test_croatian_diacritics_are_not_silently_collapsed():
    """HR aliases may record ASCII spellings, but identity keeps orthography."""
    for canonical, ascii_variant in (
        ("Možemo!", "Mozemo!"),
        ("Željana Zovko", "Zeljana Zovko"),
        ("Sunčana Glavak", "Suncana Glavak"),
        ("Božo Petrov", "Bozo Petrov"),
    ):
        assert cb.identity_key(canonical) != cb.identity_key(ascii_variant)


def test_croatia_profile_is_country_scoped_and_bilingual():
    profile = cb.COUNTRY_PROFILES["HR"]
    assert profile["country"] == "Croatia"
    assert profile["languages"] == ["hr", "en"]
    assert profile["file"] == "ep24_hr_private.json"


# --------------------------------------------------------------------------- #
# Portugal (#72): bilingual profile + Portuguese orthography stay explicit
# --------------------------------------------------------------------------- #

def test_portuguese_diacritics_are_not_silently_collapsed():
    """ASCII variants may be aliases, but not canonical identity keys."""
    for canonical, ascii_variant in (
        ("João", "Joao"),
        ("António", "Antonio"),
        ("Coligação", "Coligacao"),
        ("Cidadãos", "Cidadaos"),
    ):
        assert cb.identity_key(canonical) != cb.identity_key(ascii_variant)


def test_portugal_profile_is_country_scoped_and_bilingual():
    profile = cb.COUNTRY_PROFILES["PT"]
    assert profile["country"] == "Portugal"
    assert profile["languages"] == ["pt", "en"]
    assert profile["file"] == "ep24_pt_private.json"


def test_portuguese_short_acronyms_must_remain_country_scoped():
    """Short aliases such as PS/AD must never become cross-country identities."""
    assert _new_entry_id("PS", country="PT") != _new_entry_id("PS", country="FR")
    assert _new_entry_id("AD", country="PT") != _new_entry_id("AD", country="ES")
