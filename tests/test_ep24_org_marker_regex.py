"""Regression tests for the escaped-word-boundary defect in ``_ORG_MARKERS``.

Issue #116 follow-up. PR #120 introduced ``english_translation_required`` and the
``_ORG_MARKERS`` pattern that keeps capitalised organisation labels from being
mistaken for personal names. The pattern was written with a **double-escaped**
word boundary inside a raw string::

    re.compile(r"\\\\b(partia|...|sozialdemokraten)\\\\b")

In a raw string ``r"\\\\b"`` is a literal backslash followed by ``b``, not a word
boundary, so the alternation could never match. Everything downstream silently
mis-classified organisations as personal names, making
``english_translation_required`` return ``False`` for exactly the entries that
need translation work.

This is the same class as the earlier ``MODEL_PREFIXES`` literal-``\\n`` defect in
this repository, which is why it is pinned explicitly rather than left to the
behavioural tests alone: a boundary that cannot match fails *quietly*.

No private codebook content; synthetic fixtures only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402
from roihu_codebooks import CodebookEntry  # noqa: E402


def _entry(label: str, langs: list[str]) -> CodebookEntry:
    return CodebookEntry(entry_id="x", kind="entity", label=label, source_languages=langs)


class TestPatternIsUsable:
    """The pattern must contain real word boundaries, not escaped literals."""

    def test_pattern_has_no_double_escaped_boundary(self):
        """The defect: a literal backslash+b can never match."""
        pattern = cb._ORG_MARKERS.pattern
        assert "\\\\b" not in pattern, (
            "the pattern contains a double-escaped boundary; in a raw string this "
            "is a literal backslash + 'b' and never matches"
        )

    def test_pattern_starts_and_ends_with_a_word_boundary(self):
        pattern = cb._ORG_MARKERS.pattern
        assert pattern.startswith(r"\b"), pattern[:12]
        assert pattern.endswith(r"\b"), pattern[-12:]

    def test_pattern_matches_an_actual_organisation_word(self):
        """Proves the regex is executable, not merely well-shaped."""
        for word in ("partia", "rassemblement", "party", "moderaterna"):
            assert re.search(cb._ORG_MARKERS.pattern, word), word

    def test_pattern_does_not_match_an_unrelated_word(self):
        assert not re.search(cb._ORG_MARKERS.pattern, "Bielan")
        assert not re.search(cb._ORG_MARKERS.pattern, "Sánchez")


class TestTranslationMetricOnOrganisations:
    """The behaviour the escaping bug broke."""

    def test_non_english_organisation_needs_both(self):
        entry = _entry("Rassemblement National", ["fr"])
        assert cb.english_label_required(entry) is True
        assert cb.english_translation_required(entry) is True

    def test_organisations_are_translation_work(self):
        for label in ("Partido Socialista", "Fianna Fáil", "Sinn Féin",
                      "Les Républicains", "Moderaterna", "Bündnis 90/Die Grünen"):
            entry = _entry(label, ["fr"])
            assert cb.english_translation_required(entry) is True, label

    def test_person_name_separates_review_from_translation(self):
        """A person's English form is the same string: review, not translation."""
        entry = _entry("Adam Bielan", ["pl"])
        assert cb.english_label_required(entry) is True
        assert cb.english_translation_required(entry) is False

    def test_handles_and_urls_are_neither(self):
        for label in ("@fundacjawosp", "https://example.org"):
            entry = _entry(label, ["pl"])
            assert cb.english_translation_required(entry) is False, label

    def test_already_english_is_not_translation_work(self):
        assert cb.english_translation_required(_entry("Abortion Rights", ["en"])) is False
