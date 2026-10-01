#!/usr/bin/env python3
"""EP24 electoral-list and coalition relations (issue #74).

2024 European Parliament campaigns contain **electoral lists / coalitions that
are analytically meaningful entities distinct from their constituent parties**.
Croatia is the regression case named in the issue: several 2024 lists combined
multiple parties, so replacing a coalition mention with one of its members
silently rewrites what the source said.

This module implements the **relation layer** the issue specifies. It is public
and data-agnostic: real EP24 relations live in `LaclauGPT-Private`, and the
tests use invented fixtures only.

Design rules taken directly from the issue
------------------------------------------

1. **No silent flattening of list -> party.** A list mention resolves to the
   *list* object. Constituents stay available as related context but are never
   emitted as observed evidence.
2. **Relations are scoped and provenance-carrying.** Every relation knows its
   country, its validity window, and where it came from.
3. **The evidence firewall is explicit.** `context_for()` returns background
   relations; `observed_entities()` returns only what the source actually said.
   The two are different functions returning different things precisely so a
   caller cannot mistake one for the other.
4. **Backward compatible.** A plain party that is its own list works; a person
   with no list works; unknown ids do not raise.

What this module deliberately does NOT do
-----------------------------------------

It does not resolve surface forms. The issue's predecessor comment in #72 found
that the countries' alias coverage is ~87% empty, so a relation layer that
resolved its own endpoints would resolve almost nothing in practice. Endpoints
here are **already-canonical entry ids**; turning a surface form into one is
`roihu_identity.resolve_surface`'s job, and it abstains rather than guessing.
Keeping those apart is why this layer can be tested honestly today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Relation vocabulary
# ---------------------------------------------------------------------------
#: The relation types the issue requires. Named as constants so a typo is an
#: error at import time rather than a silently-ignored edge.
HAS_MEMBER = "electoral_list --has_member--> party"
CANDIDATE_ON = "person --candidate_on--> electoral_list"
MEMBER_OF_LIST = "party --member_of_list--> electoral_list"
EU_GROUP = "party/person --eu_group--> eu_group"

RELATION_TYPES = (HAS_MEMBER, CANDIDATE_ON, MEMBER_OF_LIST, EU_GROUP)


@dataclass(frozen=True)
class Relation:
    """One sourced, scoped, temporally-bounded edge between two entities.

    ``valid_from``/``valid_to`` are plain strings because EP24 provenance is
    heterogeneous (dates, election names, "2024-06"). Empty means "unbounded",
    which is honest for a relation nobody dated; it is not a claim of permanence.
    """

    relation: str
    source_id: str
    target_id: str
    country: str = ""
    valid_from: str = ""
    valid_to: str = ""
    provenance: str = ""
    note: str = ""

    def covers(self, period: str) -> bool:
        """Whether this relation is asserted for ``period``.

        An unbounded relation covers everything; a bounded one covers an exact
        string match on either endpoint, and a period strictly inside the window.
        String comparison is deliberate: EP24 validity windows are not all ISO
        dates, and inventing a date parser here would guess at formats the
        private data has not been shown to use.
        """
        if not period:
            return True
        if not self.valid_from and not self.valid_to:
            return True
        if period == self.valid_from or period == self.valid_to:
            return True
        return bool(self.valid_from) and bool(self.valid_to) and self.valid_from <= period <= self.valid_to


@dataclass
class RelationIndex:
    """An in-memory index over sourced relations.

    Deliberately small and dependency-free. A relation layer with no relations
    is not useful, but a relation layer whose index needs a database before it
    can be tested is not verifiable in CI, and the issue requires synthetic
    public tests.
    """

    relations: list[Relation] = field(default_factory=list)

    # -- construction ------------------------------------------------------

    def add(self, relation: Relation) -> None:
        self.relations.append(relation)

    def load(self, payload: Iterable[dict[str, Any]]) -> "RelationIndex":
        """Load relations from plain dicts (a private JSON shape, or a fixture).

        Unknown keys are ignored rather than raising: the private payload is
        authoritative and may carry fields this public layer does not model yet.
        A *malformed relation* (missing type/source/target) does raise, because
        silently dropping it would hide a real gap in the relation data.
        """
        for item in payload:
            relation = str(item.get("relation") or "").strip()
            source_id = str(item.get("source_id") or "").strip()
            target_id = str(item.get("target_id") or "").strip()
            if not relation or not source_id or not target_id:
                raise ValueError(
                    f"malformed relation (needs relation/source_id/target_id): {item!r}"
                )
            if relation not in RELATION_TYPES:
                raise ValueError(
                    f"unknown relation type {relation!r}; known: {list(RELATION_TYPES)}"
                )
            self.add(
                Relation(
                    relation=relation,
                    source_id=source_id,
                    target_id=target_id,
                    country=str(item.get("country") or "").strip().upper(),
                    valid_from=str(item.get("valid_from") or "").strip(),
                    valid_to=str(item.get("valid_to") or "").strip(),
                    provenance=str(item.get("provenance") or "").strip(),
                    note=str(item.get("note") or "").strip(),
                )
            )
        return self

    # -- queries -----------------------------------------------------------

    def _scoped(self, country: str, period: str) -> list[Relation]:
        code = country.strip().upper()
        return [
            relation
            for relation in self.relations
            if (not code or not relation.country or relation.country == code)
            and relation.covers(period)
        ]

    def members_of(self, list_id: str, *, country: str = "", period: str = "") -> list[Relation]:
        """Parties that are members of an electoral list."""
        return [
            relation
            for relation in self._scoped(country, period)
            if relation.relation == HAS_MEMBER and relation.target_id == list_id
        ]

    def lists_of(self, party_id: str, *, country: str = "", period: str = "") -> list[Relation]:
        """Lists a party stood on."""
        return [
            relation
            for relation in self._scoped(country, period)
            if relation.relation == MEMBER_OF_LIST and relation.source_id == party_id
        ]

    def candidates_of(self, list_id: str, *, country: str = "", period: str = "") -> list[Relation]:
        """People who were candidates on an electoral list."""
        return [
            relation
            for relation in self._scoped(country, period)
            if relation.relation == CANDIDATE_ON and relation.target_id == list_id
        ]

    def lists_for_candidate(self, person_id: str, *, country: str = "", period: str = "") -> list[Relation]:
        """Lists a person was a candidate on."""
        return [
            relation
            for relation in self._scoped(country, period)
            if relation.relation == CANDIDATE_ON and relation.source_id == person_id
        ]

    def eu_group_of(self, entity_id: str, *, country: str = "", period: str = "") -> list[Relation]:
        """EU-level group affiliation, only where temporally sourced."""
        return [
            relation
            for relation in self._scoped(country, period)
            if relation.relation == EU_GROUP and relation.source_id == entity_id
        ]

    def constituents(self, list_id: str, *, country: str = "", period: str = "") -> list[str]:
        """The member party ids of a list, de-duplicated and order-stable."""
        ids = [r.source_id for r in self.members_of(list_id, country=country, period=period)]
        return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# The flattening guard: observed evidence vs background relation
# ---------------------------------------------------------------------------

def observed_entities(resolved: Iterable[dict[str, Any]]) -> list[str]:
    """Canonical labels a source item actually **said**.

    Only ``EXISTING`` resolutions count. ``AMBIGUOUS``, ``CANDIDATE`` and
    ``NEW`` are excluded on purpose: a fuzzy candidate is a guess, and treating
    a guess as an observation is exactly the failure this module exists to
    prevent.
    """
    labels: list[str] = []
    for result in resolved:
        if result.get("decision") != "EXISTING":
            continue
        label = str(result.get("canonical_label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def context_for(
    list_id: str,
    index: RelationIndex,
    *,
    country: str = "",
    period: str = "",
) -> list[dict[str, str]]:
    """Background relations for a list mention — **never** evidence.

    The issue requires that RAG can "retrieve a list and its constituents
    without treating constituents as mentioned evidence". So this returns the
    constituents *labelled as background*, and the caller must not feed them to
    ``observed_entities``. ``members_are_evidence`` is hard-coded False rather
    than left to the caller to remember.
    """
    rows: list[dict[str, str]] = []
    for relation in index.members_of(list_id, country=country, period=period):
        rows.append({
            "relation": relation.relation,
            "member_id": relation.source_id,
            "list_id": list_id,
            "provenance": relation.provenance,
            "evidence_role": "background_context_not_source_evidence",
            "members_are_evidence": "false",
        })
    for relation in index.candidates_of(list_id, country=country, period=period):
        rows.append({
            "relation": relation.relation,
            "member_id": relation.source_id,
            "list_id": list_id,
            "provenance": relation.provenance,
            "evidence_role": "background_context_not_source_evidence",
            "members_are_evidence": "false",
        })
    return rows


def flattening_violations(
    observed: Iterable[dict[str, Any]],
    *,
    list_id: str,
    index: RelationIndex,
    country: str = "",
    period: str = "",
) -> list[str]:
    """Member ids that were emitted as *observed* rather than as context.

    This is the machine-checkable form of the issue's central requirement. It
    returns the offending member ids, so a caller can assert emptiness rather
    than reading prose.

    A member that genuinely appears in the source text is **not** a violation —
    the issue is explicit that the problem is a member being reported *because
    the list was mentioned*, not a member being mentioned. So this function does
    not try to decide intent; it reports members that are present in the
    observed set while the list is not, which is the fingerprint of a
    flattening step having replaced the list with one of its parts.
    """
    label_to_member: dict[str, str] = {}
    for member_id in index.constituents(list_id, country=country, period=period):
        label_to_member[member_id] = member_id

    observed_labels = set(observed_entities(observed))
    if list_id in observed_labels:
        # The list itself was observed, so members appearing alongside it is not
        # flattening — the source named both.
        return []
    return sorted(
        label
        for label in observed_labels
        if label in label_to_member
    )
