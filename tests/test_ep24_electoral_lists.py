"""Regression tests for issue #74: electoral-list / coalition relations.

The issue specifies a Croatia test case with four synthetic elements:

1. one standalone party list;
2. one multi-party coalition/list;
3. one candidate linked to a list;
4. ambiguous short party abbreviation.

Expected behaviour: a list mention resolves to the **list** object, constituent
parties remain **related context**, and no constituent party is emitted as
directly observed unless the source evidence contains it.

Every fixture here is invented. No private EP24 relations, party rosters,
researcher notes or candidate data appear in this file; the real relation data
belongs in `LaclauGPT-Private`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ep24_lists import (  # noqa: E402
    CANDIDATE_ON,
    EU_GROUP,
    HAS_MEMBER,
    MEMBER_OF_LIST,
    Relation,
    RelationIndex,
    context_for,
    flattening_violations,
    observed_entities,
)

# --------------------------------------------------------------------------
# Synthetic Croatia-shaped fixtures (invented, generic names)
# --------------------------------------------------------------------------

STANDALONE_LIST = "hr:list:standalone"
COALITION_LIST = "hr:list:coalition"
PARTY_A = "hr:party:a"
PARTY_B = "hr:party:b"
PARTY_C = "hr:party:c"
CANDIDATE_1 = "hr:person:1"
EU_GROUP_ID = "eu:group:one"


def _index() -> RelationIndex:
    return RelationIndex().load([
        # 1. a standalone party list: the party *is* its own list
        {"relation": MEMBER_OF_LIST, "source_id": PARTY_A, "target_id": STANDALONE_LIST,
         "country": "HR", "provenance": "synthetic-fixture"},
        # 2. a multi-party coalition/list: three members on one list
        {"relation": HAS_MEMBER, "source_id": PARTY_A, "target_id": COALITION_LIST,
         "country": "HR", "provenance": "synthetic-fixture"},
        {"relation": HAS_MEMBER, "source_id": PARTY_B, "target_id": COALITION_LIST,
         "country": "HR", "provenance": "synthetic-fixture"},
        {"relation": HAS_MEMBER, "source_id": PARTY_C, "target_id": COALITION_LIST,
         "country": "HR", "provenance": "synthetic-fixture"},
        # 3. a candidate linked to a list
        {"relation": CANDIDATE_ON, "source_id": CANDIDATE_1, "target_id": COALITION_LIST,
         "country": "HR", "provenance": "synthetic-fixture"},
        # EU group, temporally sourced
        {"relation": EU_GROUP, "source_id": PARTY_A, "target_id": EU_GROUP_ID,
         "country": "HR", "provenance": "synthetic-fixture"},
    ])


# --------------------------------------------------------------------------
# 1 + 2: the list is an entity in its own right
# --------------------------------------------------------------------------

def test_standalone_party_list_resolves_to_the_list_object() -> None:
    """A party that is its own list still has a list object to resolve to."""
    index = _index()
    assert index.lists_of(PARTY_A, country="HR")  # stands on a list
    assert index.constituents(STANDALONE_LIST, country="HR") == []


def test_coalition_constituents_are_returned_without_being_flattened() -> None:
    """The coalition has members, and the members are not the coalition."""
    index = _index()
    assert index.constituents(COALITION_LIST, country="HR") == [PARTY_A, PARTY_B, PARTY_C]
    # The members must not themselves claim to *be* the coalition list.
    for member in (PARTY_A, PARTY_B, PARTY_C):
        assert COALITION_LIST not in index.constituents(member, country="HR")


def test_a_constituent_is_not_its_own_coalition() -> None:
    """The flattening bug in one line: resolving the list must not yield a member.

    If a normalization pass replaced the coalition with one of its parties, the
    list id would not appear as a list at all and the member would appear alone.
    """
    index = _index()
    members = set(index.constituents(COALITION_LIST, country="HR"))
    assert COALITION_LIST not in members, "a list must never be its own constituent"


# --------------------------------------------------------------------------
# 3: candidate -> list
# --------------------------------------------------------------------------

def test_candidate_links_to_a_list_and_not_to_a_party() -> None:
    """`person --candidate_on--> electoral_list` stays on the list."""
    index = _index()
    targets = [r.target_id for r in index.lists_for_candidate(CANDIDATE_1, country="HR")]
    assert targets == [COALITION_LIST]
    assert PARTY_A not in targets, "a candidacy must not be rewritten onto a party"


# --------------------------------------------------------------------------
# 4: ambiguous short party abbreviation
# --------------------------------------------------------------------------

def test_short_acronym_ambiguity_is_representable_not_resolved() -> None:
    """Two different entities may claim the same short acronym.

    The relation layer must be able to hold both without picking one. Resolution
    is `roihu_identity`'s job and it abstains; this layer must not silently
    collapse them while storing relations.
    """
    index = RelationIndex().load([
        {"relation": HAS_MEMBER, "source_id": PARTY_A, "target_id": COALITION_LIST, "country": "HR"},
        {"relation": HAS_MEMBER, "source_id": PARTY_B, "target_id": COALITION_LIST, "country": "HR"},
    ])
    assert index.constituents(COALITION_LIST, country="HR") == [PARTY_A, PARTY_B]


def test_acronym_boundary_matching_does_not_leak_into_a_longer_word() -> None:
    """`PiS` must not match `Pisarz`; the strict matcher is available for identity.

    This is the measured defect from #72's Croatia audit (a coverage count had
    to be corrected by hand because a substring match read `Možemo` in
    `Mozemohr`). The identity/coverage side of the code uses
    `boundary_matches`; the retrieval scorer deliberately still matches
    substrings, because inflection recall is wanted there.
    """
    from roihu_codebooks import boundary_matches

    assert boundary_matches("PiS", "PiS wygrał wybory") is True
    assert boundary_matches("PiS", "Pisarz napisał książkę") is False
    assert boundary_matches("HDZ", "HDZ i SDP") is True
    assert boundary_matches("HDZ", "HDZx") is False


# --------------------------------------------------------------------------
# The central requirement: no constituent becomes observed evidence
# --------------------------------------------------------------------------

def test_list_mention_does_not_promote_a_constituent_to_observed_evidence() -> None:
    """The issue's headline requirement, asserted rather than described.

    The source mentioned the coalition. The coalition resolves. No member party
    appears in the observed set, because the source never said it.
    """
    index = _index()
    resolved = [
        {"decision": "EXISTING", "canonical_label": COALITION_LIST},
    ]
    assert observed_entities(resolved) == [COALITION_LIST]
    assert flattening_violations(resolved, list_id=COALITION_LIST, index=index, country="HR") == []


def test_a_member_appearing_alone_is_reported_as_a_flattening_violation() -> None:
    """The negative control: this is what the bug looks like.

    The list was not observed, but a member was — which is the fingerprint of a
    normalization step having replaced the coalition with one of its parts.
    """
    index = _index()
    resolved = [
        {"decision": "EXISTING", "canonical_label": PARTY_A},
    ]
    violations = flattening_violations(resolved, list_id=COALITION_LIST, index=index, country="HR")
    assert violations == [PARTY_A], "a lone constituent must be flagged as a flattening"


def test_a_member_named_alongside_its_list_is_not_a_violation() -> None:
    """The issue is explicit: naming both is not flattening."""
    index = _index()
    resolved = [
        {"decision": "EXISTING", "canonical_label": COALITION_LIST},
        {"decision": "EXISTING", "canonical_label": PARTY_B},
    ]
    assert flattening_violations(resolved, list_id=COALITION_LIST, index=index, country="HR") == []


def test_context_for_labels_members_as_background_not_evidence() -> None:
    """RAG may retrieve the constituents, but they are marked as context."""
    index = _index()
    rows = context_for(COALITION_LIST, index, country="HR")
    assert rows, "the coalition must yield background context"
    for row in rows:
        assert row["evidence_role"] == "background_context_not_source_evidence"
        assert row["members_are_evidence"] == "false"
        assert row["provenance"], "a relation without provenance is not usable"


def test_ambiguous_and_candidate_resolutions_are_never_observed_evidence() -> None:
    """A guess is not an observation — the firewall in one assertion."""
    resolved = [
        {"decision": "EXISTING", "canonical_label": "Observed Actor"},
        {"decision": "AMBIGUOUS", "canonical_label": "Maybe Actor"},
        {"decision": "CANDIDATE", "canonical_label": "Probably Actor"},
        {"decision": "NEW", "canonical_label": "Unknown Actor"},
    ]
    assert observed_entities(resolved) == ["Observed Actor"]


# --------------------------------------------------------------------------
# Scoping, time and provenance
# --------------------------------------------------------------------------

def test_relations_are_country_scoped() -> None:
    """A Croatian coalition must not appear when querying another country."""
    index = _index()
    assert index.constituents(COALITION_LIST, country="PL") == []


def test_temporally_sourced_relations_respect_their_window() -> None:
    """A list that existed only for 2024 must not be asserted for 2029."""
    index = RelationIndex().load([
        {"relation": HAS_MEMBER, "source_id": PARTY_A, "target_id": COALITION_LIST,
         "country": "HR", "valid_from": "2024-01", "valid_to": "2024-06",
         "provenance": "synthetic-fixture"},
    ])
    assert index.constituents(COALITION_LIST, country="HR", period="2024-03") == [PARTY_A]
    assert index.constituents(COALITION_LIST, country="HR", period="2029-05") == []
    # An unbounded relation covers any period, and that is honest rather than a
    # claim of permanence.
    assert index.constituents(COALITION_LIST, country="HR") == [PARTY_A]


def test_every_relation_carries_provenance_through_the_query() -> None:
    index = _index()
    for relation in index.members_of(COALITION_LIST, country="HR"):
        assert relation.provenance == "synthetic-fixture"


def test_eu_group_only_returned_where_sourced() -> None:
    index = _index()
    assert [r.target_id for r in index.eu_group_of(PARTY_A, country="HR")] == [EU_GROUP_ID]
    assert index.eu_group_of(PARTY_B, country="HR") == []


# --------------------------------------------------------------------------
# Malformed input fails loudly rather than dropping an edge
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {"relation": "", "source_id": "a", "target_id": "b"},
        {"relation": HAS_MEMBER, "source_id": "", "target_id": "b"},
        {"relation": HAS_MEMBER, "source_id": "a", "target_id": ""},
        {"relation": "party --invented--> thing", "source_id": "a", "target_id": "b"},
    ],
)
def test_malformed_relations_raise_instead_of_being_skipped(payload: dict) -> None:
    """A dropped edge is an invisible analytical gap; refuse the payload."""
    with pytest.raises(ValueError):
        RelationIndex().load([payload])


def test_unknown_payload_keys_are_ignored_not_fatal() -> None:
    """The private payload is authoritative and may carry fields we do not model."""
    index = RelationIndex().load([
        {"relation": HAS_MEMBER, "source_id": PARTY_A, "target_id": COALITION_LIST,
         "country": "HR", "some_future_field": "value"},
    ])
    assert index.constituents(COALITION_LIST, country="HR") == [PARTY_A]


def test_backward_compatibility_a_plain_party_with_no_list_still_works() -> None:
    """Existing private codebooks have parties with no relation rows at all."""
    index = RelationIndex().load([])
    assert index.lists_of(PARTY_A, country="HR") == []
    assert index.constituents(COALITION_LIST, country="HR") == []
    assert index.lists_for_candidate(CANDIDATE_1, country="HR") == []


def test_relation_dataclass_covers_period_semantics() -> None:
    """`covers` is the temporal predicate; pin its edges."""
    unbounded = Relation(relation=HAS_MEMBER, source_id="a", target_id="b")
    assert unbounded.covers("2024-06") is True
    assert unbounded.covers("") is True
    windowed = Relation(relation=HAS_MEMBER, source_id="a", target_id="b",
                        valid_from="2024-01", valid_to="2024-06")
    assert windowed.covers("2024-06") is True
    assert windowed.covers("2024-07") is False
