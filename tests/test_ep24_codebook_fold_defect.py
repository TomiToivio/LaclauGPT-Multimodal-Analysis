"""Regression tests for the EP24 codebook diacritic-folding defect (issue #72).

These pin the behaviour of ``scripts/ep24/fold_fix.py`` and, importantly, assert
that the fix does **not** over-reach: Polish ``ł`` is a distinct letter (/w/), so
collapsing it to ``l`` for identity would be a false merge, not a normalisation win.

The France pass (#91) extended this file with the part the Polish pass left
undone: the auditor, ``scripts/ep24/codebook_coverage.py``, shipped the fix but
never applied it, and the blast-radius scan only covered a Latin block. Both are
now pinned, because both had published wrong numbers.

No private data is used. All fixtures are synthetic or public entity names.
"""

import importlib.util
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
    non_decomposing_scripts,
)

# The auditor is a script, not an installed module, so load it by path.
_SPEC = importlib.util.spec_from_file_location(
    "codebook_coverage", ROOT / "scripts" / "ep24" / "codebook_coverage.py"
)
assert _SPEC and _SPEC.loader
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


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

    def test_french_ligatures_are_in_the_non_decomposing_set(self):
        """The France pass (#91): œ/æ are ordinary French vocabulary, not a corner case."""
        letters = non_decomposing_letters()
        for char in ("œ", "Œ", "æ", "Æ"):
            assert char in letters, char
        assert fold_broken("cœur") != fold_fixed("cœur")

    def test_the_scan_covers_cyrillic_not_only_latin(self):
        """The scan must cover every script, or it cannot report the worst case.

        The Bulgarian pass found the old Latin-only scan (`U+0100`–`U+024F`)
        silent about Cyrillic even though the defect deleted *every* Bulgarian
        capital letter and drove 52 of 316 BG labels to the empty string.
        """
        scripts = non_decomposing_scripts()
        assert "Cyrillic" in scripts, f"the scan must reach Cyrillic; got {sorted(scripts)[:8]}"
        assert scripts["Cyrillic"] > 50, f"expected the whole Cyrillic block, got {scripts['Cyrillic']}"
        assert "Latin" in scripts
        # Sanity: a Cyrillic letter that survives NFKD and was therefore deleted.
        assert "\u0401" not in non_decomposing_letters(), "Ё decomposes (Ё -> Е + breve)"
        assert "\u0413" in non_decomposing_letters(), "Г (Cyrillic GHE) does not decompose"

    def test_cyrillic_labels_were_deleted_by_the_broken_fold(self):
        """The concrete consequence in the BG book: whole labels folded to ""."""
        for label in ("ГЕРБ", "Бойко Борисов", "Движение за права и свободи", "Възраждане"):
            assert fold_broken(label) == "", f"{label!r} should vanish under the old fold"
            assert fold_fixed(label) != "", f"{label!r} must survive the corrected fold"


class TestAuditorUsesTheFix:
    """The PL pass shipped ``fold_fix.py`` and never wired it in. Pin the wiring.

    Until this was fixed, ``scripts/ep24/codebook_coverage.py`` kept its own copy
    of the broken fold, so the auditor and the regression tests disagreed about
    what folding means — and the published cross-country table in
    ``docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md`` was computed with the broken
    one. These tests fail if the local copy is ever reintroduced.
    """

    def test_the_auditor_fold_is_the_corrected_fold(self):
        for probe in ("Arłukowicz", "cœur", "ГЕРБ", "Żukowska", "Kraków"):
            assert coverage._fold(probe) == fold_fixed(probe), probe

    def test_the_auditor_does_not_reintroduce_the_deletion_filter(self):
        """The old implementation is gone: no letter may be silently dropped."""
        for probe in ("ГЕРБ", "cœur", "Arłukowicz"):
            assert coverage._fold(probe).strip() != "", probe

    def test_bulgarian_fragmentation_is_reproducible_with_the_shipped_broken_fold(self):
        """The published BG figure cannot be reproduced — this is what that means.

        The cross-country table records BG fragmentation as 34. The auditor as
        published yields 32; with the corrected fold it yields 37. 34 is neither,
        because 52 BG labels were invisible to grouping, so the number moved for
        reasons unrelated to any codebook change.

        This test asserts the *ordering*, not the exact number: the corrected
        fold must see more fragmentation than the broken one, because Cyrillic
        labels stop being dropped.
        """
        entries = [
            {"kind": "entity", "label": "Продължаваме промяната"},
            {"kind": "entity", "label": "Продължаваме промяната – Демократична България"},
            {"kind": "entity", "label": "ГЕРБ"},
            {"kind": "entity", "label": "ГЕРБ – СДС"},
        ]
        with_fixed = _group_count(entries, fold_fixed)
        with_broken = _group_count(entries, fold_broken)
        assert with_broken == 0, "with every Cyrillic label folded to '', nothing can group"
        assert with_fixed > with_broken


def _group_count(entries, fold) -> int:
    """Reproduce the auditor's fragmentation grouping with a given fold."""
    groups: dict[str, set[str]] = {}
    for entry in entries:
        key = fold(entry["label"]).split()
        key = [t for t in key if len(t) > 3]
        if not key:
            continue
        groups.setdefault(key[-1], set()).add(entry["label"])
    return sum(1 for members in groups.values() if len(members) > 1)
