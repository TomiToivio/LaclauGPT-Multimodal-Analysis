from __future__ import annotations

from roihu_codebooks import (
    CodebookEntry,
    english_label_required,
    english_translation_required,
)


def _entry(label: str, langs: list[str]) -> CodebookEntry:
    return CodebookEntry(
        entry_id="x",
        kind="entity",
        label=label,
        source_languages=langs,
    )


def test_person_name_separates_review_from_translation_work() -> None:
    entry = _entry("Adam Bielan", ["pl"])
    assert english_label_required(entry) is True
    assert english_translation_required(entry) is False


def test_non_english_organisation_needs_both() -> None:
    entry = _entry("Rassemblement National", ["fr"])
    assert english_label_required(entry) is True
    assert english_translation_required(entry) is True


def test_already_english_unknown_language_needs_neither() -> None:
    entry = _entry("Abortion", [])
    assert english_label_required(entry) is False
    assert english_translation_required(entry) is False
