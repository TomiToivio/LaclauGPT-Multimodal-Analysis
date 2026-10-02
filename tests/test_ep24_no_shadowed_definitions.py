#!/usr/bin/env python3
"""Regression tests: no module-level name may be defined twice (#116).

Root cause of #116
------------------
`roihu_codebooks.py` accumulated two blocks of English-label policy — #110's
"needs a gloss only when the *label* is not already English" and #108/#114's
"every non-English-sourced entry needs an explicit label". Each block defined its
own helpers, and three names collided:

    _NON_ENGLISH_MARKERS        line 326   (label policy)   shadowed by line 451
    _looks_like_person_name     line 345   (label policy)   shadowed by line 462
    label_looks_english         line 380   (label policy)   shadowed by line 485

Python keeps the last definition, so `entry_needs_english_label` — written
against the *label* policy's helpers, and defined *above* the shadowing block —
silently executed the *language* policy's marker list. Nothing in CI looks for
this, which is why 509 tests passed: the shadowing was self-consistent.

These tests pin the two things that make the defect unrepeatable:

1. **No name is defined twice at module level.** A mechanical check, so a third
   policy block cannot be added by simply appending to the file.
2. **Both vocabularies are explicitly reachable and named**, so which one decides
   an entry is a visible argument rather than an artifact of definition order.

This file is PUBLIC and needs no private material.

Run:  python -m pytest tests/test_ep24_no_shadowed_definitions.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402
from roihu_codebooks import CodebookEntry  # noqa: E402

MODULE = ROOT / "roihu_codebooks.py"


def _module_level_definitions(path: Path) -> dict[str, list[int]]:
    """Every module-level ``def``/``class``/assignment target -> line numbers.

    Only *bare* module scope (``node.col_offset == 0``) counts. Names defined
    inside a function or a conditional block are legitimate and are skipped —
    the defect being pinned is two top-level definitions of one name.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, list[int]] = {}

    def record(name: str, lineno: int) -> None:
        found.setdefault(name, []).append(lineno)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    record(target.id, node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # Only if it has a value: `X: int` alone declares without defining.
            if node.value is not None:
                record(node.target.id, node.lineno)
    return found


def test_no_module_level_name_is_defined_twice() -> None:
    """The exact defect from #116, mechanically: no duplicate top-level name.

    This is the guard that would have caught it. It is deliberately generic —
    it knows nothing about English-label policy — because the failure mode is
    "someone appended a second implementation of an existing name", which is a
    general hazard on a repo several agents edit in parallel.
    """
    defs = _module_level_definitions(MODULE)
    duplicates = {name: lines for name, lines in defs.items() if len(lines) > 1}
    assert not duplicates, (
        "module-level name(s) defined more than once in roihu_codebooks.py; the "
        f"last definition silently wins: {duplicates}"
    )


def test_the_three_names_from_the_reported_defect_are_resolved() -> None:
    """The specific #116 collisions: each is now either single or renamed away.

    `_NON_ENGLISH_MARKERS` and `_looks_like_person_name` were the private names
    that collided; they are now policy-qualified, so the old bare names must NOT
    exist (keeping them would reintroduce exactly the ambiguity that caused the
    bug). `label_looks_english` was public and still is, exactly once.
    """
    defs = _module_level_definitions(MODULE)
    # renamed away: a bare private name here would be ambiguous again
    assert defs.get("_NON_ENGLISH_MARKERS", []) == []
    assert defs.get("_looks_like_person_name", []) == []
    # the public name remains, defined exactly once
    assert len(defs.get("label_looks_english", [])) == 1, defs.get("label_looks_english")
    # and both policies' names exist exactly once each
    for name in (
        "_LABEL_POLICY_NON_ENGLISH_MARKERS",
        "_LANGUAGE_POLICY_NON_ENGLISH_MARKERS",
        "_label_policy_person_name",
        "_language_policy_person_name",
        "_label_policy_looks_english",
        "_language_policy_looks_english",
    ):
        assert len(defs.get(name, [])) == 1, f"{name}: {defs.get(name)}"


def test_the_two_policy_vocabularies_have_distinct_names() -> None:
    """Each design's helpers must be reachable under their own name.

    Distinct names are what make the vocabularies separable at all; before #116
    the label policy's helpers were unreachable dead code.
    """
    for name in (
        "_label_policy_person_name",
        "_label_policy_looks_english",
        "_LABEL_POLICY_NON_ENGLISH_MARKERS",
        "_language_policy_person_name",
        "_language_policy_looks_english",
        "_LANGUAGE_POLICY_NON_ENGLISH_MARKERS",
    ):
        assert hasattr(cb, name), f"{name} missing — a policy's helpers are unreachable"


def test_the_two_marker_lists_are_actually_different() -> None:
    """The vocabularies are not interchangeable, which is why this matters.

    If the two lists were identical the shadowing would have been harmless. They
    are not: the label policy lists ``koalicion``/``alliance``/``momentum``/``dk``,
    the language policy lists ``partidos``/``républicains``/``coalicion``.
    """
    label_pat = cb._LABEL_POLICY_NON_ENGLISH_MARKERS.pattern
    lang_pat = cb._LANGUAGE_POLICY_NON_ENGLISH_MARKERS.pattern
    assert label_pat != lang_pat
    # one marker unique to each side
    assert "koalicion" in label_pat and "koalicion" not in lang_pat
    assert "publicains" in lang_pat and "publicains" not in label_pat


# --------------------------------------------------------------------------- #
# the live wiring is explicit, and the alternative is reachable
# --------------------------------------------------------------------------- #


def _entry(label: str, langs: list[str] | None = None) -> CodebookEntry:
    return CodebookEntry(entry_id="x", kind="entity", label=label, source_languages=langs or ["pl"])


def test_vocabulary_is_an_explicit_argument_not_a_definition_order_artefact() -> None:
    """Both vocabularies can be selected by name, and they can disagree.

    `Koalicion` is the demonstration case: the label policy's marker list flags it
    as non-English, the language policy's list does not. Before #116 the second
    silently applied, so the label policy's answer was unreachable.
    """
    entry = _entry("Koalicion")
    via_language = cb.entry_needs_english_label_via_label_vocabulary(entry, vocabulary=cb.LANGUAGE_POLICY_VOCABULARY)
    via_label = cb.entry_needs_english_label_via_label_vocabulary(entry, vocabulary=cb.LABEL_POLICY_VOCABULARY)
    assert via_language is False
    assert via_label is True
    assert via_language != via_label, "if these agree the vocabularies converged; revisit #116"


def test_live_behaviour_matches_the_language_policy_vocabulary() -> None:
    """`entry_needs_english_label` must keep the behaviour that shipped.

    It resolved to the language policy's helpers before #116 (they were the
    surviving definitions). Picking the label policy's vocabulary is a
    methodology change and `AGENTS.md` §38/§22 reserve that for the author, so the
    default must be the merged behaviour until it is chosen deliberately.
    """
    for label in ("Koalicion", "Bloc", "Venstre", "Nowoczesna", "Rassemblement National", "Abortion"):
        entry = _entry(label)
        assert cb.entry_needs_english_label(entry) == cb.entry_needs_english_label_via_label_vocabulary(
            entry, vocabulary=cb.LANGUAGE_POLICY_VOCABULARY
        ), label


def test_an_unknown_vocabulary_is_rejected_not_defaulted() -> None:
    """A typo must not silently fall back to a policy."""
    import pytest

    with pytest.raises(ValueError):
        cb.entry_needs_english_label_via_label_vocabulary(_entry("Abortion"), vocabulary="labell")


def test_the_public_alias_still_exists_for_existing_callers() -> None:
    """`label_looks_english` was public before #116 and a test uses it."""
    assert callable(cb.label_looks_english)
    assert cb.label_looks_english("Abortion Rights") is True
    assert cb.label_looks_english("Rassemblement National") is False


def test_the_public_alias_resolves_to_the_language_policy_vocabulary() -> None:
    """`label_looks_english` must keep answering with the vocabulary it always did.

    Before #116 the second (language-policy) definition won, so that is the
    answer external callers observed. Repointing the alias at the label policy
    would be a silent behaviour change for every existing caller — and it would
    pass a test that only checks the two labels on which the policies agree.

    `Partidos` is the distinguishing case: the language policy's marker list
    contains ``partidos?`` and the label policy's does not, so the two predicates
    disagree on it. Asserting the alias against BOTH predicates is what makes a
    repoint fail here rather than ship.
    """
    assert cb.label_looks_english is cb._language_policy_looks_english
    assert cb.label_looks_english is not cb._label_policy_looks_english
    # and the disagreement is real, so the identity check above is load-bearing
    assert cb._language_policy_looks_english("Partidos") is False
    assert cb._label_policy_looks_english("Partidos") is True
    assert cb.label_looks_english("Partidos") is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
