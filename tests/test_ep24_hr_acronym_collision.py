#!/usr/bin/env python3
"""Croatia (HR) re-check findings: a real acronym collision the audits missed.

Issue #91 / #72, Croatia pass 3. This file is PUBLIC: synthetic fixtures only,
no private codebook content, rows, notes or personal data.

Why this file exists
--------------------
The HR audit material records that the build guard drops every Croatian party
acronym (``HDZ``, ``SDP``, ``DP``, ``MOST``…) and that recovering them is
"country/kind-scoped exact-match aliases" work (docs/ep24_country_audits/HR.md,
pass 2 addendum). That framing assumes scoping by *country* is sufficient.

It is not, for Croatia. On the official 2024 EP ballot **two different actors
share the abbreviation ``PiP``**:

1. ``PRAVO I PRAVDA`` — the party, ballot abbreviation ``PiP``, led by Mislav
   Kolakušić (0 seats in 2024).
2. ``ISTARSKA STRANKA UMIROVLJENIKA – PARTITO ISTRIANO DEI PENSIONATI –
   ISU – PIP`` — an abbreviation that appears inside the *name* of a partner on
   the IDS-led "Fair Play List 9" (0 seats in 2024).

Source: DIP (Državno izborno povjerenstvo) official results, 10 June 2024 —
https://www.izbori.hr/site/UserDocsImages/2024/Izbori_za_EU_parlament/EU%20parlament%202024.%20Rezultati%20izbora.pdf
Both strings appear verbatim in the published list of candidate lists.

So a country-scoped exact-match alias is *still* ambiguous inside HR, and the
retrieval path resolves it by returning both entries with score 1.0. Those tests
pin the current behaviour so the collision is visible and does not regress
silently into "one of them wins".

Scope note: these tests use only public electoral facts. They assert the
*matcher's* behaviour; they do not assert which party a given mention means, and
they do not touch the private codebook.

Run:  python -m pytest tests/test_ep24_hr_acronym_collision.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import (  # noqa: E402
    CodebookEntry,
    boundary_matches,
    context_block,
    identity_key,
    score_entry,
)
from roihu_identity import resolve_surface  # noqa: E402


def _entry(label: str, aliases: list[str], *, etype: str = "party") -> CodebookEntry:
    return CodebookEntry(
        entry_id=f"HR:{label}",
        kind="entity",
        label=label,
        aliases=list(aliases),
        country="HR",
        source_languages=["hr"],
        english_label="",
        definition="",
        english_definition="",
        disambiguation="",
        entity_type=etype,
        review_state="canonical",
        origin="researcher",
        locked=True,
        valid_from="",
        valid_to="",
        layer="country",
        sources=[],
        metadata={},
    )


# The two real HR actors that both carry "PiP" on the 2024 EP ballot.
PIP_PARTY = _entry("Pravo i Pravda", ["PiP"])
PIP_PENSIONERS = _entry(
    "Istarska stranka umirovljenika - Partito Istriano dei Pensionati", ["ISU", "PiP"]
)
HR_ENTRIES = [PIP_PARTY, PIP_PENSIONERS]
AMBIGUOUS_QUERY = "PiP je osvojio glasove na izborima"


def test_pip_is_genuinely_ambiguous_within_one_country() -> None:
    """HR-01: `PiP` matches two distinct HR entries — scoping by country is not enough.

    This is the finding. Both entries score a full match, so "scope the acronym
    by country" does not disambiguate Croatia's ``PiP``; the collision is
    *inside* the country.
    """
    hits = [e.label for e in HR_ENTRIES if score_entry(AMBIGUOUS_QUERY, e) == 1.0]
    assert len(hits) == 2, (
        "if this is no longer 2, either the collision disappeared from the data "
        "or the matcher started preferring one entry — both need a look"
    )


def test_ambiguous_acronym_retrieval_returns_both_rather_than_guessing() -> None:
    """HR-02: the retrieval path surfaces BOTH entries instead of silently picking one.

    Returning both is the safe outcome: background context is advisory, so
    showing two candidates is a quality problem, not a correctness one. What
    must never happen is one entry being dropped by an ordering accident.
    """
    _block, prov = context_block(AMBIGUOUS_QUERY, HR_ENTRIES, country="HR", language="hr")
    labels = {s["label"] for s in prov["selected"]}
    assert labels == {PIP_PARTY.label, PIP_PENSIONERS.label}


def test_ambiguous_acronym_context_is_never_source_evidence() -> None:
    """HR-03: the evidence firewall holds even for a colliding acronym.

    The collision is exactly the situation where a model is most likely to
    assert "the item mentions PiP the party". The context packet must still
    label itself background-only.
    """
    _block, prov = context_block(AMBIGUOUS_QUERY, HR_ENTRIES, country="HR", language="hr")
    assert prov["evidence_role"] == "background_context_not_source_evidence"


def test_identity_resolution_abstains_on_the_colliding_acronym() -> None:
    """HR-04: the identity path returns AMBIGUOUS, not an arbitrary winner.

    `roihu_identity` is the path that decides what an actor *is*, so it is the
    one that must abstain. If this ever returns exactly one entry, a coin-flip
    has been introduced into identity.
    """
    result = resolve_surface("PiP", HR_ENTRIES, country="HR", kinds=["entity"])
    assert result["decision"] == "AMBIGUOUS", (
        f"expected AMBIGUOUS for a colliding acronym, got {result['decision']!r}"
    )


def test_boundary_matcher_refuses_the_substring_hazard_on_hr_forms() -> None:
    """HR-05: strict matching rejects the HR substring traps; substring matching hits them.

    `Most` is both a Croatian party and the first four letters of `Mostar`; `HDZ`
    matches `HDZx`. This is the measured reason identity/coverage must not use
    the retrieval scorer's substring rule.
    """
    assert boundary_matches("Most", "Mostar") is False
    assert boundary_matches("HDZ", "HDZx") is False
    # Controls: the acronym as a real word, and inside Croatian prose.
    assert boundary_matches("HDZ", "HDZ je pobijedio") is True
    assert boundary_matches("Most", "Most je osvojio") is True


def test_diacritics_are_load_bearing_in_croatian_identity() -> None:
    """HR-06: `Možemo` and a diacritic-stripped `Mozemo` are NOT the same identity key.

    The HR audit forbids diacritic stripping in the canonical identity key. The
    runtime already agrees — `identity_key` keeps `ž` — which is worth pinning
    because the *auditor's* fold strips it, so the two layers would disagree if
    anyone wired the auditor's fold into identity.
    """
    assert identity_key("Možemo") != identity_key("Mozemo")
    assert identity_key("Možemo") == identity_key("Možemo")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
