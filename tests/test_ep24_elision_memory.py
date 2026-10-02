#!/usr/bin/env python3
"""Regression tests: the elision fold must not create two CANONICAL objects (#102).

The FR audit measured the damage at the codebook layer, but the sharper
consequence is here: two spellings of one entity became two *canonical memory
objects*, each individually valid, nothing flagging it:

    'Besoin d'Europe'  -> A-1b23218f83f451b5
    'Besoin d’Europe'  -> A-9eb91724e91f80b9

The fix recognises a spelling that differs ONLY by the elision apostrophe and
attaches it as an alias of the existing object — without recomputing either id,
so nothing already stored moves. This file pins that, plus the scoping the fold
must not breach.

This file is PUBLIC: synthetic fixtures only, no private data.

Run:  python -m pytest tests/test_ep24_elision_memory.py -v
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402
import roihu_memory as rm  # noqa: E402

ASCII = "\u0027"  # '
RIGHT = "\u2019"  # ’


def _memory(tmp_path: Path) -> rm.EP24Memory:
    return rm.EP24Memory(tmp_path / "memory.sqlite3")


# --------------------------------------------------------------------------- #
# the defect
# --------------------------------------------------------------------------- #


def test_two_spellings_would_have_been_two_objects(tmp_path: Path) -> None:
    """MEM-01: without recognition the two spellings are two distinct ids.

    `stable_id` is unchanged (it still uses the un-folded `surface_key`), so the
    two ids genuinely differ — that is the migration constraint, and it is why
    recognition rather than re-keying is the fix.
    """
    a = rm.stable_id("actor", f"Besoin d{ASCII}Europe", country="FR", language="fr")
    b = rm.stable_id("actor", f"Besoin d{RIGHT}Europe", country="FR", language="fr")
    assert a != b


def test_surface_key_is_unchanged_and_the_fold_is_separate() -> None:
    """MEM-02: the fold is a separate key; `surface_key` (the id seed) is not touched."""
    a, b = f"Besoin d{ASCII}Europe", f"Besoin d{RIGHT}Europe"
    assert rm.surface_key(a) != rm.surface_key(b)
    assert rm.elision_surface_key(a) == rm.elision_surface_key(b)
    # and the fold converges on the project's ASCII apostrophe
    assert rm.elision_surface_key(b) == rm.surface_key(a)


# --------------------------------------------------------------------------- #
# recognition: one identity instead of two CANONICAL objects
# --------------------------------------------------------------------------- #


def test_the_second_spelling_does_not_create_a_second_object(tmp_path: Path) -> None:
    """MEM-03: the fix for the FR audit's two-canonical-objects outcome.

    Adding the variant spelling returns the SAME object id as the first spelling,
    so one entity stays one canonical object. This is the behaviour that was
    missing: before, each spelling produced its own valid, unflagged object.
    """
    mem = _memory(tmp_path)
    first = mem.add_object("actor", f"Besoin d{ASCII}Europe", country="FR", language="fr", state="CANONICAL")
    second = mem.add_object("actor", f"Besoin d{RIGHT}Europe", country="FR", language="fr", state="CANONICAL")
    assert first == second, "an apostrophe variant must not mint a second object"

    with mem.connect() as db:
        n = db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    assert n == 1, "one entity, one object"


def test_the_variant_still_resolves_to_that_object(tmp_path: Path) -> None:
    """MEM-03b: and the variant spelling is still resolvable afterwards."""
    mem = _memory(tmp_path)
    obj = mem.add_object("actor", f"Besoin d{ASCII}Europe", country="FR", language="fr", state="CANONICAL")
    mem.add_object("actor", f"Besoin d{RIGHT}Europe", country="FR", language="fr", state="CANONICAL")
    resolved = mem.resolve(f"Besoin d{RIGHT}Europe", "actor", country="FR", language="fr")
    assert resolved.decision == "EXISTING"
    assert resolved.obj_id == obj


def test_a_variant_spelling_does_not_resolve_as_new(tmp_path: Path) -> None:
    """MEM-04: NEW would invite a second canonical object; it must not be NEW."""
    mem = _memory(tmp_path)
    mem.add_object("actor", f"l{ASCII}Union europeenne", country="FR", language="fr", state="CANONICAL")
    resolved = mem.resolve(f"l{RIGHT}Union europeenne", "actor", country="FR", language="fr")
    assert resolved.decision != "NEW"


def test_the_convergence_is_recorded_not_silent(tmp_path: Path) -> None:
    """MEM-05: the acceptance criterion — migration behaviour documented and tested."""
    mem = _memory(tmp_path)
    a = mem.add_object("actor", f"Besoin d{ASCII}Europe", country="FR", language="fr", state="CANONICAL")
    mem.add_object("actor", f"Besoin d{RIGHT}Europe", country="FR", language="fr", state="CANONICAL")

    merges = mem.elision_alias_merges()
    assert merges, "an elision convergence must be recorded"
    assert any(m["kept_id"] == a for m in merges)
    assert all(m["reason"] == "elision apostrophe variant" for m in merges)


def test_no_id_moves_when_recognition_happens(tmp_path: Path) -> None:
    """MEM-06: the id a caller gets is the un-folded one, before and after.

    The whole point of the design: recognition adds an alias, it does not
    re-key. If this ever changes, every stored id is at risk.
    """
    mem = _memory(tmp_path)
    typographic = f"Besoin d{RIGHT}Europe"
    expected = "A-" + hashlib.sha256(
        "|".join(("actor", rm.surface_key(typographic), "FR", "fr", "")).encode()
    ).hexdigest()[:16]
    got = mem.add_object("actor", typographic, country="FR", language="fr", state="CANONICAL")
    assert got == expected


# --------------------------------------------------------------------------- #
# isolation the fold must not breach
# --------------------------------------------------------------------------- #


def test_elision_recognition_is_kind_scoped(tmp_path: Path) -> None:
    """MEM-07: an actor must not resolve to a topic that folds the same way.

    The query must use a *typographic* apostrophe and the stored object the ASCII
    one (or vice versa), so the fold is actually exercised rather than
    short-circuiting on an already-canonical query. The stored topic carries the
    typographic form and the actor query the ASCII form, with both in country FR:
    if kind scoping were dropped from the twin search, the actor query would
    resolve to the topic object.
    """
    mem = _memory(tmp_path)
    topic = mem.add_object("topic", f"l{RIGHT}Europe sociale", country="FR", language="fr", state="CANONICAL")
    resolved = mem.resolve(f"l{ASCII}Europe sociale", "actor", country="FR", language="fr")
    assert resolved.decision == "NEW", "kind isolation must hold across the fold"
    assert resolved.obj_id != topic


def test_elision_recognition_is_country_scoped(tmp_path: Path) -> None:
    """MEM-08: the same folded label in another country is a different entity.

    Stored in DE with the typographic apostrophe, queried in FR with the ASCII
    one — same folded key, different country. Both directions of the fold are
    exercised, so dropping the country check in the twin search fails this.
    """
    mem = _memory(tmp_path)
    other = mem.add_object("actor", f"l{RIGHT}Europe", country="DE", language="de", state="CANONICAL")
    resolved = mem.resolve(f"l{ASCII}Europe", "actor", country="FR", language="fr")
    assert resolved.decision == "NEW", "country isolation must hold across the fold"
    assert resolved.obj_id != other


def test_deprecated_objects_are_not_resurrected_by_the_fold(tmp_path: Path) -> None:
    """MEM-09: `accepted_only` must still gate the fold, as it gates exact lookup.

    The query uses the OTHER apostrophe from the stored object so the fold path is
    genuinely reached; a query in the same form would short-circuit on the
    exact-alias branch and this would pass for the wrong reason.
    """
    mem = _memory(tmp_path)
    obj = mem.add_object("actor", f"l{RIGHT}Europe", country="FR", language="fr", state="CANONICAL")
    mem.set_state(obj, "DEPRECATED")
    assert mem.resolve(f"l{ASCII}Europe", "actor", country="FR", language="fr").decision == "NEW"
    # with accepted_only=False it is visible again, exactly like an exact alias hit
    assert mem.resolve(f"l{ASCII}Europe", "actor", country="FR", language="fr", accepted_only=False).decision == "EXISTING"


# --------------------------------------------------------------------------- #
# the fold must not become a fuzzy matcher
# --------------------------------------------------------------------------- #


def test_a_different_entity_is_not_recognised_by_the_fold(tmp_path: Path) -> None:
    """MEM-10: only apostrophe differences count. Different words must stay NEW."""
    mem = _memory(tmp_path)
    mem.add_object("actor", f"Besoin d{ASCII}Europe", country="FR", language="fr", state="CANONICAL")
    assert mem.resolve("Besoin de France", "actor", country="FR", language="fr").decision == "NEW"
    assert mem.resolve("Besoin Europe", "actor", country="FR", language="fr").decision == "NEW"
    assert mem.resolve(f"Pas besoin d{RIGHT}Europe", "actor", country="FR", language="fr").decision == "NEW"


def test_prime_is_not_treated_as_an_elision(tmp_path: Path) -> None:
    """MEM-11: the not-an-apostrophe boundary holds on the memory side too.

    Uses a typographic apostrophe in the query against a stored prime form (a
    prime has no elision variant, so the fold path is reached and must decline).
    A query in the same form as the stored object would short-circuit and prove
    nothing.
    """
    mem = _memory(tmp_path)
    mem.add_object("actor", "5\u20325", country="FR", language="fr", state="CANONICAL")
    assert mem.resolve(f"5{RIGHT}5", "actor", country="FR", language="fr").decision == "NEW"
    # and the fold must not equate a prime with an apostrophe at the key level
    assert rm.elision_surface_key("5\u20325") != rm.elision_surface_key(f"5{ASCII}5")


def test_the_two_modules_agree_on_the_folded_family() -> None:
    """MEM-12: the memory and codebook folds must not drift apart.

    `roihu_memory` deliberately does not import the codebook layer, so the
    constant is mirrored. This pins the mirror: if one list gains a character and
    the other does not, identity and memory would disagree about elision.
    """
    assert rm.ELISION_APOSTROPHES == cb.ELISION_APOSTROPHES
    assert rm.PROJECT_APOSTROPHE == cb.PROJECT_APOSTROPHE


def test_elision_alias_merges_is_empty_for_a_clean_store(tmp_path: Path) -> None:
    """A store with no apostrophe variants records no convergence noise."""
    mem = _memory(tmp_path)
    mem.add_object("actor", "Rassemblement National", country="FR", language="fr", state="CANONICAL")
    assert mem.elision_alias_merges() == []


def test_recognition_is_idempotent(tmp_path: Path) -> None:
    """Re-adding the same variant must not multiply rows."""
    mem = _memory(tmp_path)
    mem.add_object("actor", f"l{ASCII}Europe", country="FR", language="fr", state="CANONICAL")
    for _ in range(3):
        mem.add_object("actor", f"l{RIGHT}Europe", country="FR", language="fr", state="CANONICAL")
    assert len(mem.elision_alias_merges()) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
