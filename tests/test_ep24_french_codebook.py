"""France-specific EP24 codebook regressions for issue #91.

Synthetic fixtures only — no private codebook content, rows or researcher notes.

The French defects are *typographic* rather than orthographic, so these tests pin
three separate things that the other country passes did not have to cover:

1. ``œ``/``æ`` are single letters, not ``o``+``e`` / ``a``+``e``. NFKD does not
   decompose them, so a fold built on "strip non-ASCII" destroys them and the
   word with them. French uses these in ordinary vocabulary (``cœur``, ``œuvre``),
   and ``scripts/ep24/fold_fix.py`` must keep the letter rather than delete it.
2. French elision is written with **either** the ASCII apostrophe (``l'Union``,
   what keyboards and ASR produce) **or** U+2019 (``l’Union``, what word
   processors and much French news web output produce). ``identity_key``
   preserves punctuation on purpose, so these are two identities. The tests
   document that, so nobody "fixes" it by making the identity key fuzzy.
3. French party identification runs through two- and three-letter acronyms
   (``RN``, ``LR``, ``PS``, ``LFI``, ``NFP``, ``EELV``, ``PCF``), so the codebook
   must carry them as **aliases of a long canonical label**, not as labels that
   glue the acronym inside parentheses — ``Rassemblement National (RN)`` cannot be
   reached from a post that says only ``le RN``.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import (  # noqa: E402
    COUNTRY_PROFILES,
    context_block,
    identity_key,
    load_codebook,
)
from scripts.ep24.apostrophe_hygiene import normalise_elision  # noqa: E402
from scripts.ep24.fold_fix import fold_broken, fold_fixed  # noqa: E402


def _book(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "ep24_fr_private.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "FR",
                "country": "France",
                "language": "fr",
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _labels(query: str, entries, country: str = "FR") -> list[str]:
    _block, provenance = context_block(query, entries, country=country, language="fr")
    return [row["label"] for row in provenance["selected"]]


# --------------------------------------------------------------------------
# 1. The FR profile and its French-specific letters
# --------------------------------------------------------------------------


def test_fr_profile_is_bilingual_and_uses_expected_private_file() -> None:
    profile = COUNTRY_PROFILES["FR"]
    assert profile["country"] == "France"
    assert profile["languages"] == ["fr", "en"]
    assert profile["file"] == "ep24_fr_private.json"


def test_french_accents_are_not_destructively_folded_in_identity() -> None:
    """`É`/`é` are distinct letters. Identity must not silently ASCII them away."""
    assert identity_key("Élysée") == identity_key("éLYSÉE")
    assert identity_key("Élysée") != identity_key("Elysee")
    assert identity_key("Mélenchon") != identity_key("Melenchon")
    assert identity_key("Hayer") == identity_key("hayer")


def test_ligature_letters_are_not_decomposed_by_nfkd() -> None:
    """The root cause, stated so a regression is unmistakable."""
    assert unicodedata.normalize("NFKD", "œ") == "œ"
    assert unicodedata.normalize("NFKD", "æ") == "æ"
    # Contrast: these DO decompose, which is why French accents fold and these
    # two do not.
    assert unicodedata.normalize("NFKD", "é") != "é"
    assert unicodedata.normalize("NFKD", "î") != "î"


# --------------------------------------------------------------------------
# 2. The fold must not destroy French words
# --------------------------------------------------------------------------


class TestFrenchFold:
    """`cœur` is one word. A fold that returns `c ur` has broken the corpus."""

    def test_the_defect_is_real(self) -> None:
        """Documents the defect so the fix cannot silently regress.

        ``œ``/``æ`` are ligature *letters*, not base+diacritic, so NFKD leaves
        them alone and the old ``[^a-z0-9 ]`` filter then deleted them. The
        damage is worse than one lost character: the word is split, so a
        downstream token count sees two tokens where the source had one.
        """
        assert fold_broken("cœur") == "c ur"
        assert fold_broken("œuvre") == "uvre", "a leading ligature is deleted outright"
        assert fold_broken("sœur") == "s ur"
        assert fold_broken("fœtus") == "f tus"

    @pytest.mark.parametrize(
        "word",
        ["cœur", "œuvre", "manœuvre", "mœurs", "sœur", "œuf", "fœtus", "næss"],
    )
    def test_fixed_fold_keeps_the_ligature_and_the_token(self, word: str) -> None:
        fixed = fold_fixed(word)
        assert fixed == word, "the ligature letter must survive the fold"
        assert " " not in fixed, "one word must not become two tokens"
        assert fixed, "the token must not be deleted"

    def test_fixed_fold_still_folds_decomposing_french_accents(self) -> None:
        """The useful half of the original behaviour has to survive."""
        assert fold_fixed("Élysée") == "elysee"
        assert fold_fixed("député") == "depute"
        assert fold_fixed("François-Xavier") == "francois xavier"
        assert fold_fixed("Raphaël") == "raphael"

    def test_ligatures_are_not_merged_onto_two_letter_sequences(self) -> None:
        """`œ` and `oe` are different strings — do not silently merge them.

        Both spellings are attested in real French text (`cœur` and `coeur`), so
        this is a review decision with provenance, not a fold.
        """
        assert fold_fixed("cœur") != fold_fixed("coeur")
        assert fold_fixed("œuvre") != fold_fixed("oeuvre")


# --------------------------------------------------------------------------
# 3. The apostrophe: two characters, one elision
# --------------------------------------------------------------------------


class TestFrenchApostrophe:
    def test_nfc_does_not_unify_the_two_apostrophes(self) -> None:
        """The characters are canonically unrelated, so NFC cannot repair them.

        This is what makes the defect silent: no normalising pass in the project
        can know which apostrophe was meant.
        """
        assert unicodedata.normalize("NFC", "l'Union") != unicodedata.normalize("NFC", "l\u2019Union")
        assert unicodedata.category("'") == "Po"
        assert unicodedata.category("\u2019") == "Pf"

    def test_identity_key_keeps_the_two_apostrophes_distinct(self) -> None:
        """Documents current behaviour — punctuation is preserved ON PURPOSE.

        Accents and punctuation are analytically meaningful (HR audit rule 1), so
        this is a property to preserve, not a bug to fix by making the identity
        key fuzzier.
        """
        assert identity_key("l'Union") != identity_key("l\u2019Union")
        assert identity_key("Besoin d'Europe") != identity_key("Besoin d\u2019Europe")

    def test_elision_normalisation_collapses_only_the_apostrophe_family(self) -> None:
        """The safe repair is narrow: apostrophe variants, nothing else."""
        assert normalise_elision("l\u2019Union") == "l'Union"
        assert normalise_elision("Besoin d\u2019Europe") == "Besoin d'Europe"
        assert identity_key(normalise_elision("l\u2019Union")) == identity_key("l'Union")

        # A prime is a different character with a different meaning. It must NOT
        # be folded onto an apostrophe.
        assert normalise_elision("5\u2032") == "5\u2032"
        # Nor may the normaliser touch accents.
        assert normalise_elision("Élysée") == "Élysée"

    def test_a_typographic_entity_is_reachable_after_elision_normalisation(self, tmp_path: Path) -> None:
        """The practical consequence: one entry, both spellings reach it."""
        entries, _meta = load_codebook(
            _book(
                tmp_path,
                [
                    {
                        "kind": "entity",
                        "label": "Besoin d'Europe",
                        "aliases": ["Besoin d’Europe"],
                        "entity_type": "electoral_list",
                    }
                ],
            )
        )
        assert _labels("La liste Besoin d’Europe", entries) == ["Besoin d'Europe"]

    def test_the_split_bites_identity_not_the_retrieval_scorer(self, tmp_path: Path) -> None:
        """Where the split actually damages the pipeline — measured, not assumed.

        It is tempting to file the apostrophe split as "the codebook can't be
        reached". That is wrong for the *scorer*: ``score_entry``'s token fallback
        splits on non-word characters, so the two apostrophes already collapse
        there and a bare label still matches. Measured:

            score_entry("Besoin d’Europe", entry labelled "Besoin d'Europe") == 1.0

        The damage is one layer down, at IDENTITY, which is what the pipeline keys
        on: `identity_key`, the derived `CB-…` / `A-…` object ids, and therefore
        the merge, both produce two objects for one entity. That is the defect
        worth fixing, and it is why the fix belongs in normalisation rather than
        in the matcher.
        """
        from roihu_codebooks import score_entry

        ascii_entry = _book(
            tmp_path,
            [{"kind": "entity", "label": "Besoin d'Europe", "aliases": [], "entity_type": "electoral_list"}],
        )
        entries, _meta = load_codebook(ascii_entry)
        # The scorer already bridges the apostrophes — do not "fix" what works.
        assert score_entry("Besoin d’Europe", entries[0]) == 1.0
        # Identity does not, and identity is what ids and merges are built on.
        assert identity_key("Besoin d'Europe") != identity_key("Besoin d’Europe")

    def test_two_canonical_memory_objects_are_created_for_one_entity(self, tmp_path: Path) -> None:
        """Was the downstream cost; now pins that #102 removed it.

        Before #102, two spellings of one French list became two CANONICAL memory
        objects with different ids and nothing flagged it, because each spelling was
        individually valid. That is the defect this test originally recorded, and its
        own docstring said: "if this ever passes with first == second, normalisation
        was fixed and the split is gone".

        #102 fixed it at the recognition layer: the variant spelling is attached as
        an alias of the existing object and the SAME id comes back, so one entity is
        one object. The assertion is inverted here to pin the fixed behaviour, and
        the id-stability half is kept, because the fix must not have moved any id.
        """
        from roihu_memory import EP24Memory, stable_id

        memory = EP24Memory(tmp_path / "memory.sqlite3")
        first = memory.add_object(
            "actor", "Besoin d'Europe", country="FR", language="fr",
            state="CANONICAL", origin="researcher_private", locked=True,
        )
        second = memory.add_object(
            "actor", "Besoin d’Europe", country="FR", language="fr",
            state="CANONICAL", origin="researcher_private", locked=True,
        )
        assert first and second
        assert first == second, (
            "one entity must be one canonical object (#102); the apostrophe variant "
            "is recognised and attached as an alias, not minted as a second object"
        )
        # Id stability: the surviving id is the un-folded one, so nothing stored
        # before #102 was re-pointed.
        assert first == stable_id("actor", "Besoin d'Europe", country="FR", language="fr")
        # and the converged pair is recorded rather than silent
        merges = memory.elision_alias_merges()
        assert merges, "the elision convergence must be recorded"


# --------------------------------------------------------------------------
# 4. Acronyms are aliases of a long label, never parenthetical labels
# --------------------------------------------------------------------------

PARTIES = [
    {
        "kind": "entity",
        "label": "Rassemblement National",
        "english_label": "National Rally",
        "aliases": ["RN", "Rassemblement national", "Front National", "FN"],
        "entity_type": "party",
    },
    {
        "kind": "entity",
        "label": "Nouveau Front populaire",
        "english_label": "New Popular Front",
        "aliases": ["NFP", "Front populaire"],
        "entity_type": "electoral_list",
    },
    {
        "kind": "entity",
        "label": "La France insoumise",
        "english_label": "France Unbowed",
        "aliases": ["LFI", "France Insoumise"],
        "entity_type": "party",
    },
    {
        "kind": "entity",
        "label": "Les Républicains",
        "english_label": "The Republicans",
        "aliases": ["LR"],
        "entity_type": "party",
    },
    {
        "kind": "entity",
        "label": "Parti socialiste",
        "english_label": "Socialist Party",
        "aliases": ["PS"],
        "entity_type": "party",
    },
]


class TestFrenchAcronyms:
    def test_fr_acronyms_retrieve_their_canonical_party(self, tmp_path: Path) -> None:
        entries, _meta = load_codebook(_book(tmp_path, PARTIES))
        assert "Rassemblement National" in _labels("le RN est en tête", entries)
        assert "Nouveau Front populaire" in _labels("le NFP a gagné", entries)
        assert "La France insoumise" in _labels("LFI dénonce", entries)
        assert "Parti socialiste" in _labels("le PS soutient", entries)

    def test_a_two_letter_fr_acronym_does_not_match_inside_a_word(self, tmp_path: Path) -> None:
        """`PS` is word-bounded in the scorer, so `apside`/`options` do not fire it.

        This is the half of the short-alias problem that already works and must
        not regress: the `< 3` branch of ``score_entry`` uses a word-boundary
        regex, so a 2-letter alias is matched as a word and never as a substring.
        """
        entries, _meta = load_codebook(_book(tmp_path, PARTIES))
        assert _labels("apside et eclipse", entries) == []
        assert _labels("le parti socialiste", entries) == ["Parti socialiste"]

    def test_short_aliases_are_inert_but_the_token_fallback_still_fires(self, tmp_path: Path) -> None:
        """The measured caveat: word-bounding helps, but the token fallback is looser.

        `PS` itself is safe. The residual hit is *not* the alias — it comes from
        `score_entry`'s token overlap, which uses tokens > 2 chars with no
        stopword list. A French label that begins with an article
        (`Les Républicains`, `Les Écologistes`, `La France insoumise`) therefore
        accumulates the token `les`, and any French sentence containing `les`
        scores against it.

        Measured on the real 650-entry FR book: 8 of 8 neutral French probes
        retrieved context, 27 spurious selections in total, all attributable to
        this. The fix is a stopword-aware token filter (or per-language stopword
        lists) in the shared scorer — a cross-country change, so it is recorded
        as a finding and tracked separately rather than patched here.
        """
        entries, _meta = load_codebook(_book(tmp_path, PARTIES))
        selected = _labels("les options possibles", entries)
        assert "Les Républicains" in selected, (
            "expected the stopword-token false positive to still be present; if "
            "this fails the scorer gained stopword filtering and the finding in "
            "docs/ep24_country_audits/FR.md can be closed"
        )
        # The acronym itself is NOT the cause: a probe with no stopword token hit
        # selects nothing.
        assert _labels("apside et eclipse", entries) == []

    def test_fr_acronyms_do_not_cross_the_country_boundary(self, tmp_path: Path) -> None:
        """`LR`/`PS` collide across countries, so the alias must stay scoped."""
        entries, _meta = load_codebook(_book(tmp_path, PARTIES))
        assert _labels("le RN est en tête", entries, country="DE") == []
        assert _labels("le PS soutient", entries, country="ES") == []

    def test_a_parenthetical_acronym_label_is_unreachable(self, tmp_path: Path) -> None:
        """The real defect: the correct acronym never appears as a *form*.

        `Rassemblement National (RN)` is the observed codebook shape. The acronym
        is glued inside parentheses, so no `forms` entry equals `RN` and a post
        saying `le RN` matches nothing. This test exists so the shape cannot be
        reintroduced as a "convenience".
        """
        entries, _meta = load_codebook(
            _book(
                tmp_path,
                [
                    {
                        "kind": "entity",
                        "label": "Rassemblement National (RN)",
                        "aliases": [],
                        "entity_type": "party",
                    }
                ],
            )
        )
        assert _labels("le RN est en tête", entries) == []
        assert "RN" not in entries[0].forms

    def test_french_local_and_english_names_are_forms_of_one_entity(self, tmp_path: Path) -> None:
        entries, _meta = load_codebook(_book(tmp_path, PARTIES))
        rn = next(entry for entry in entries if entry.label == "Rassemblement National")
        assert "National Rally" in rn.forms
        assert "Rassemblement national" in rn.forms

        block, _prov = context_block("le RN", entries, country="FR", language="fr")
        assert "Rassemblement National / National Rally" in block


# --------------------------------------------------------------------------
# 5. French electoral lists and their constituent parties stay separate
# --------------------------------------------------------------------------


def test_a_coalition_and_its_member_party_are_distinct_entries(tmp_path: Path) -> None:
    """`Besoin d'Europe` is a list; `Renaissance` is a party inside it.

    Flattening one into the other loses the list-level actor that the campaign
    actually talked about (issue #74 shape, French instance).
    """
    entries, _meta = load_codebook(
        _book(
            tmp_path,
            [
                {
                    "kind": "entity",
                    "label": "Besoin d'Europe",
                    "aliases": ["Besoin d'Europe (liste)"],
                    "entity_type": "electoral_list",
                },
                {
                    "kind": "entity",
                    "label": "Renaissance",
                    "aliases": ["Renaissance (parti)"],
                    "entity_type": "party",
                },
            ],
        )
    )
    by_label = {entry.label: entry for entry in entries}
    assert set(by_label) == {"Besoin d'Europe", "Renaissance"}
    assert by_label["Besoin d'Europe"].entry_id != by_label["Renaissance"].entry_id


def test_a_person_is_not_normalised_into_their_party(tmp_path: Path) -> None:
    """A candidate mention is not evidence that the party is mentioned."""
    entries, _meta = load_codebook(
        _book(
            tmp_path,
            [
                {
                    "kind": "entity",
                    "label": "Jordan Bardella",
                    "aliases": ["Bardella"],
                    "entity_type": "person",
                },
                {
                    "kind": "entity",
                    "label": "Rassemblement National",
                    "aliases": ["RN"],
                    "entity_type": "party",
                },
            ],
        )
    )
    selected = _labels("Bardella prend la parole", entries)
    assert "Jordan Bardella" in selected
    assert "Rassemblement National" not in selected
