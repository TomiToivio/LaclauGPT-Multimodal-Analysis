"""Remaining issue #95 requirements after the core short-boundary fix."""
from roihu_codebooks import CodebookEntry, context_block, score_entry


def _entry(
    label: str,
    aliases: list[str],
    *,
    language: str = "sv",
    metadata: dict | None = None,
    reviewed: bool = False,
):
    return CodebookEntry(
        entry_id=f"SE:{label}",
        kind="entity",
        label=label,
        aliases=aliases,
        country="SE",
        source_languages=[language],
        entity_type="party",
        review_state="CANONICAL" if reviewed else "PROVISIONAL",
        locked=reviewed,
        metadata=metadata or {},
    )


def test_declared_short_alias_is_explicit_even_on_provisional_entry():
    entry = _entry("Socialdemokraterna", ["S"])
    assert score_entry("S presenterar sin EU-politik", entry, language="sv") == 1.0


def test_unreviewed_short_label_is_not_implicitly_promoted():
    entry = _entry("S", [])
    assert score_entry("S presenterar sin EU-politik", entry, language="sv") == 0.0

    entry.metadata["reviewed_short_aliases"] = ["S"]
    assert score_entry("S presenterar sin EU-politik", entry, language="sv") == 1.0


def test_short_alias_can_be_language_scoped():
    entry = _entry(
        "Socialdemokraterna",
        ["S"],
        metadata={"short_alias_languages": {"S": ["sv"]}},
    )
    assert score_entry("S presenterar sin EU-politik", entry, language="sv") == 1.0
    assert score_entry("S presents its EU policy", entry, language="en") == 0.0


def test_short_alias_can_be_election_scoped():
    entry = _entry(
        "Socialdemokraterna",
        ["S"],
        metadata={"short_alias_elections": {"S": ["EP2024-SE"]}},
    )
    assert score_entry("S presenterar sin EU-politik", entry, election="EP2024-SE") == 1.0
    assert score_entry("S presenterar sin EU-politik", entry, election="RIX2026-SE") == 0.0


def test_context_passes_language_and_election_scope_and_stays_background_only():
    entry = _entry(
        "Socialdemokraterna",
        ["S"],
        metadata={
            "short_alias_languages": {"S": ["sv"]},
            "short_alias_elections": {"S": ["EP2024-SE"]},
        },
    )
    block, provenance = context_block(
        "S presenterar sin EU-politik",
        [entry],
        country="SE",
        language="sv",
        election="EP2024-SE",
    )
    assert "Socialdemokraterna" in block
    assert provenance["election"] == "EP2024-SE"
    # Assert the contract, not a literal version. This string has been renamed
    # once per retrieval revision (v3 -> v4 -> v5), and pinning the literal made
    # an unrelated merge (#112) turn a passing behavioural test red while every
    # assertion about actual behaviour still held. The prefix identifies the
    # method family; the suffix is free to change with the implementation.
    method = provenance["selection_method"]
    assert method.startswith("deterministic_lexical_v"), method
    assert "short_alias" in method, method
    assert provenance["evidence_role"] == "background_context_not_source_evidence"

    blocked, _ = context_block(
        "S presenterar sin EU-politik",
        [entry],
        country="SE",
        language="en",
        election="EP2024-SE",
    )
    assert blocked == ""
