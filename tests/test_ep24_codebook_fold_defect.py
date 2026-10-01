"""Regression tests for the EP24 codebook diacritic-folding defect (issue #72).

These pin the behaviour of ``scripts/ep24/fold_fix.py`` and, importantly, assert
that the fix does **not** over-reach: Polish ``ł`` is a distinct letter (/w/), so
collapsing it to ``l`` for identity would be a false merge, not a normalisation win.

No private data is used. All fixtures are synthetic or public entity names.
"""

import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ep24.fold_fix import (  # noqa: E402
    fold_broken,
    fold_fixed,
    non_decomposing_letters,
)


class TestBugIsReal:
    """Documents the defect so a regression is caught if it returns."""

    def test_polish_l_is_not_decomposed(self):
        """The root cause: ł has no NFD/NFKD decomposition."""
        assert unicodedata.normalize("NFKD", "ł") == "ł"
        assert unicodedata.normalize("NFKD", "ż") != "ż"

    def test_broken_fold_splits_a_word_containing_l_stroke(self):
        """The observed damage: one token becomes two."""
        assert fold_broken("Arłukowicz") == "ar ukowicz"

    def test_broken_fold_deletes_a_leading_l_stroke(self):
        assert fold_broken("Łępkowska") == "epkowska"

    def test_decomposing_accents_still_fold_in_the_broken_version(self):
        """Contrast: ż/ó/ń DO fold, which is why the bug hid for other languages."""
        assert fold_broken("Żukowska") == "zukowska"
        assert fold_broken("Kraków") == "krakow"


class TestFix:
    """The corrected behaviour."""

    @pytest.mark.parametrize(
        "label",
        ["Arłukowicz", "Łępkowska", "Bożena Przyłuska", "Ewa Zajączkowska-Hernik"],
    )
    def test_fixed_fold_never_splits_a_word(self, label):
        fixed = fold_fixed(label)
        # token count must be preserved (ignoring genuinely hyphenated parts)
        assert "  " not in fixed
        for token in label.replace("-", " ").split():
            norm = fold_fixed(token)
            assert norm in fixed.replace(" ", ""), (label, norm, fixed)

    def test_fixed_fold_preserves_the_l_stroke_letter(self):
        assert "ł" in fold_fixed("Arłukowicz")
        assert "ł" in fold_fixed("Łępkowska")

    def test_fixed_fold_still_folds_real_accents(self):
        """The useful part of the original behaviour must survive."""
        assert fold_fixed("Żukowska") == "zukowska"
        assert fold_fixed("Kraków") == "krakow"
        assert fold_fixed("Kamińska") == "kaminska"

    def test_fixed_fold_does_not_merge_the_two_letters(self):
        """Guards the false-merge direction: ł and l are NOT the same letter."""
        assert fold_fixed("Arłukowicz") != fold_fixed("Arlukowicz")
        assert fold_fixed("Łępkowska") != fold_fixed("Lepkowska")

    def test_fixed_fold_is_idempotent(self):
        for label in ("Arłukowicz", "Kraków", "Żukowska", "Anna Maria Żukowska"):
            once = fold_fixed(label)
            assert fold_fixed(once) == once


class TestBlastRadius:
    """The defect is language-specific, which is why it distorted the comparison."""

    def test_l_stroke_is_in_the_non_decomposing_set(self):
        assert "ł" in non_decomposing_letters()
        assert "Ł" in non_decomposing_letters() or "ł" in non_decomposing_letters()

    def test_croatian_portuguese_spanish_are_unaffected(self):
        """Explains why only PL showed the inflated fragmentation count."""
        for label in ("Šefčovič", "João", "González", "Müller", "Åkesson"):
            assert fold_broken(label) == fold_fixed(label), label
