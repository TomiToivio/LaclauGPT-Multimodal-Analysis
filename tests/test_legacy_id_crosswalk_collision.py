"""Issue #11 §1 known-issue 2 — a label-only upstream id must not silently merge.

`upstream_stable_id()` derives its key from `upstream_legacy_key()`, which strips
accents and parenthesised qualifiers. That is deliberate legacy compatibility, so
the hash itself must not change: existing ids are already in use.

The consequence, however, was a false merge. Two genuinely distinct objects can
claim the same upstream id, and a lookup returned BOTH with no indication that the
id was ambiguous -- i.e. it resolved by insertion order.

These tests pin the fix: the legacy hash is byte-compatible, the collision is
recorded, and the read side reports ambiguity rather than picking a winner.
"""
from __future__ import annotations

from pathlib import Path

from roihu_memory import (
    EP24Memory,
    stable_id,
    surface_key,
    upstream_legacy_key,
    upstream_stable_id,
)


def _memory(tmp_path: Path) -> EP24Memory:
    return EP24Memory(tmp_path / "memory.sqlite3")


# --- the legacy hash must not move -------------------------------------------


def test_upstream_hash_is_unchanged_for_legacy_compatibility():
    """The conflation is deliberate; changing the hash would move existing ids."""
    assert upstream_stable_id("actor", "Puolue (A)") == upstream_stable_id("actor", "Puolue (B)")
    assert upstream_stable_id("actor", "Grüne") == upstream_stable_id("actor", "Grune")
    # the historically-asserted equivalence is preserved exactly
    assert upstream_stable_id("actor", "Åsa (MEP)") == upstream_stable_id("actor", "Asa")


def test_context_aware_id_still_distinguishes_them():
    """The context-aware id is the thing that keeps the two objects apart."""
    a = stable_id("actor", "Puolue (A)", country="FI", language="fi")
    b = stable_id("actor", "Puolue (B)", country="FI", language="fi")
    assert a != b


# --- the collision is now recorded -------------------------------------------


def test_two_distinct_objects_on_one_upstream_id_are_both_kept(tmp_path):
    mem = _memory(tmp_path)
    a = mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
    b = mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")
    assert a != b

    up = upstream_stable_id("actor", "Puolue (A)")
    resolved = mem.resolve_upstream_id(up)
    assert set(resolved["obj_ids"]) == {a, b}, "both claimants must be retained"


def test_the_shared_upstream_id_is_reported_ambiguous(tmp_path):
    mem = _memory(tmp_path)
    a = mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
    mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")

    up = upstream_stable_id("actor", "Puolue (A)")
    before = mem.resolve_upstream_id(up)
    assert before["ambiguous"] is True
    assert len(before["obj_ids"]) == 2

    # a genuinely unique id must NOT be flagged
    solo = mem.add_object("actor", "Unique Solo Party", country="FI", language="fi", state="CANONICAL")
    single = mem.resolve_upstream_id(upstream_stable_id("actor", "Unique Solo Party"))
    assert single["ambiguous"] is False
    assert single["obj_ids"] == [solo]
    assert a in before["obj_ids"]


def test_collision_is_recorded_for_operator_review(tmp_path):
    mem = _memory(tmp_path)
    a = mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
    b = mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")

    collisions = mem.crosswalk_collisions()
    assert collisions, "the collision must be recorded, not silent"
    up = upstream_stable_id("actor", "Puolue (A)")
    claimants = {c["claimant_id"] for c in collisions if c["upstream_id"] == up}
    assert claimants == {a, b}
    for c in collisions:
        assert c["reason"]


def test_a_unique_upstream_id_records_no_collision(tmp_path):
    mem = _memory(tmp_path)
    mem.add_object("actor", "Only One Party", country="FI", language="fi", state="CANONICAL")
    assert mem.crosswalk_collisions() == []


def test_collision_recording_is_idempotent(tmp_path):
    """Re-running an import must not multiply collision rows."""
    mem = _memory(tmp_path)
    for _ in range(3):
        mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
        mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")
    first = len(mem.crosswalk_collisions())
    mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
    mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")
    assert len(mem.crosswalk_collisions()) == first


# --- accent / qualifier cases named in the issue ------------------------------


def test_accent_variants_are_flagged_not_silently_merged(tmp_path):
    """`Grüne` vs `Grune` is the accent case the issue names."""
    mem = _memory(tmp_path)
    a = mem.add_object("actor", "Grüne", country="DE", language="de", state="CANONICAL")
    b = mem.add_object("actor", "Grune", country="DE", language="de", state="CANONICAL")
    up = upstream_stable_id("actor", "Grüne")
    resolved = mem.resolve_upstream_id(up)
    assert resolved["ambiguous"] is True
    assert set(resolved["obj_ids"]) >= {a} or set(resolved["obj_ids"]) >= {b}


def test_unknown_upstream_id_is_empty_not_an_error(tmp_path):
    mem = _memory(tmp_path)
    resolved = mem.resolve_upstream_id("A-does-not-exist")
    assert resolved["obj_ids"] == []
    assert resolved["ambiguous"] is False


# --- the distinction the issue asks us to preserve ---------------------------


def test_raw_forms_are_preserved_on_the_objects(tmp_path):
    """The objects keep their full labels; only the legacy key is lossy."""
    mem = _memory(tmp_path)
    mem.add_object("actor", "Puolue (A)", country="FI", language="fi", state="CANONICAL")
    mem.add_object("actor", "Puolue (B)", country="FI", language="fi", state="CANONICAL")
    with mem.connect() as db:
        labels = {row[0] for row in db.execute("SELECT canonical_label FROM objects")}
    assert {"Puolue (A)", "Puolue (B)"} <= labels
    # and the surface key (which keeps them distinct) is what aliases use
    assert surface_key("Puolue (A)") != surface_key("Puolue (B)")
    assert upstream_legacy_key("Puolue (A)") == upstream_legacy_key("Puolue (B)")
