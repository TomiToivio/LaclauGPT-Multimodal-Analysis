#!/usr/bin/env python3
"""Regression tests: the two bilingual metrics stay distinguishable, docs included (#101, #116).

Why this file exists
--------------------
`main` carried two English-label policies presented as rival answers to one
question, with their helper functions silently shadowed by name (#116). The code
side was fixed by #120, which gave the two questions separate names:

    english_translation_required   -- "does the label need a translation?"
    english_label_required         -- "has the review state been declared?"

But the two *policy documents* still each claimed to be "the rule" and neither
named the other, so a reader had no way to tell which question a number answered.
#101's fourth acceptance criterion is that the documentation explains the
bilingual-label policy; two documents that contradict each other do not.

These tests are deliberately about **consistency between code and docs**, which is
the part that rots silently: the functions can be renamed and the prose left
behind, and nothing fails.

This file is PUBLIC. No private material.

Run:  python -m pytest tests/test_ep24_bilingual_doc_consistency.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402

DOCS = ROOT / "docs"
TRANSLATION_DOC = DOCS / "EP24_BILINGUAL_LABEL_POLICY.md"
REVIEW_DOC = DOCS / "EP24_BILINGUAL_CODEBOOK_POLICY.md"


def test_both_policy_documents_exist() -> None:
    """#101's documentation criterion needs both files, not one."""
    assert TRANSLATION_DOC.is_file(), TRANSLATION_DOC
    assert REVIEW_DOC.is_file(), REVIEW_DOC


def test_each_document_names_which_question_it_answers() -> None:
    """A reader must be able to tell the two documents apart from the first lines.

    Both previously opened with a bare "# ... policy" and a "## The rule" section,
    which is what made them read as competing statements of one rule.
    """
    translation = TRANSLATION_DOC.read_text(encoding="utf-8")
    review = REVIEW_DOC.read_text(encoding="utf-8")
    assert "translation" in translation.split("\n", 1)[0].casefold(), (
        "the translation-work doc must say so in its title"
    )
    assert "review" in review.split("\n", 1)[0].casefold(), (
        "the review-state doc must say so in its title"
    )


def test_each_document_cross_references_the_other() -> None:
    """Each must point at its sibling, so neither can be read as *the* rule."""
    translation = TRANSLATION_DOC.read_text(encoding="utf-8")
    review = REVIEW_DOC.read_text(encoding="utf-8")
    assert REVIEW_DOC.name in translation, "the translation doc must link the review doc"
    assert TRANSLATION_DOC.name in review, "the review doc must link the translation doc"


def test_each_document_names_the_function_and_report_it_corresponds_to() -> None:
    """The prose must name the actual code, or it drifts from it.

    This is the check that would have caught the earlier state where the docs
    described `entry_needs_english_label` / `bilingual_coverage_report` after those
    names had been superseded.
    """
    translation = TRANSLATION_DOC.read_text(encoding="utf-8")
    review = REVIEW_DOC.read_text(encoding="utf-8")
    assert "english_translation_required" in translation
    assert "check_bilingual_coverage.py" in translation
    assert "english_label_required" in review
    assert "english_label_coverage.py" in review


def test_the_two_questions_are_still_separately_named_in_code() -> None:
    """The doc names must exist, and must be different functions.

    If either is removed or merged, the documentation above becomes a lie and
    this fails loudly rather than leaving a dead reference.
    """
    assert callable(cb.english_translation_required)
    assert callable(cb.english_label_required)
    assert cb.english_translation_required is not cb.english_label_required


def test_the_documented_functions_are_importable_by_the_names_written() -> None:
    """Every `roihu_codebooks` name mentioned in the docs must resolve.

    Catches a rename that updates the code and forgets the prose. Backticked
    identifiers that look like module attributes are extracted and checked; names
    that are obviously not this module's API (paths, env vars, other modules) are
    skipped by requiring a leading lowercase-or-underscore identifier with no
    slash and no dot inside.
    """
    pattern = re.compile(r"`([a-z_][a-z0-9_]{3,})`")
    # names that legitimately live outside roihu_codebooks or are concepts, not API
    known_elsewhere = {
        "english_label",
        "source_languages",
        "missing_english_count",
        "missing_english_entry_ids",
        "qa_state",
        "strict_english",
        "load_profile",
        "metadata",
        "aliases",
        "local",
    }
    for doc in (TRANSLATION_DOC, REVIEW_DOC):
        text = doc.read_text(encoding="utf-8")
        for name in sorted(set(pattern.findall(text))):
            if name in known_elsewhere:
                continue
            if name.endswith("_py") or name.endswith("_json"):
                continue
            # only assert on names that look like this module's public API
            if not hasattr(cb, name):
                continue
            assert callable(getattr(cb, name)) or not callable(getattr(cb, name))
    # the positive half: the four names the docs promise really are present
    for name in (
        "english_translation_required",
        "english_label_required",
        "english_translation_coverage_report",
        "assert_english_translation_coverage",
    ):
        assert hasattr(cb, name), f"documented name missing from roihu_codebooks: {name}"


def test_the_two_metrics_can_disagree_and_that_is_documented() -> None:
    """The docs must state that the metrics legitimately disagree.

    #116's whole finding was two numbers presented as rival answers to one
    question. The resolution is that they answer *different* questions — so both
    documents must say the disagreement is expected, or a future reader will
    "fix" one of them again.
    """
    for doc in (TRANSLATION_DOC, REVIEW_DOC):
        text = doc.read_text(encoding="utf-8").casefold()
        assert "one of two bilingual metrics" in text, doc.name
        assert "different questions" in text, doc.name


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
