from ep24_relations import (
    BACKGROUND_EVIDENCE_ROLE,
    OBSERVED_EVIDENCE_ROLE,
    EP24RelationStore,
    Relation,
    relation_context,
)

def test_list_mention_keeps_list_as_evidence_and_members_as_context(tmp_path):
    store = EP24RelationStore(tmp_path / "relations.sqlite")
    store.add(Relation(
        "LIST-COALITION", "has_member", "PARTY-A",
        country="HR", election="EP2024", valid_from="2024-04-01",
        valid_to="2024-06-30", source_ref="synthetic://coalition",
        provenance="synthetic_public_fixture",
    ))
    store.add(Relation(
        "LIST-COALITION", "has_member", "PARTY-B",
        country="HR", election="EP2024", source_ref="synthetic://coalition",
        provenance="synthetic_public_fixture",
    ))
    labels = {
        "LIST-COALITION": "Alliance List",
        "PARTY-A": "Party Alpha",
        "PARTY-B": "Party Beta",
    }
    ctx = relation_context(
        ["LIST-COALITION"], store, labels=labels, country="HR", election="EP2024"
    )
    assert [item["obj_id"] for item in ctx["observed"]] == ["LIST-COALITION"]
    assert ctx["observed"][0]["evidence_role"] == OBSERVED_EVIDENCE_ROLE
    assert {item["target_id"] for item in ctx["related"]} == {"PARTY-A", "PARTY-B"}
    assert all(item["evidence_role"] == BACKGROUND_EVIDENCE_ROLE for item in ctx["related"])
    assert "must never be emitted as directly observed evidence" in ctx["firewall"]

def test_candidate_relation_is_temporally_and_election_scoped(tmp_path):
    store = EP24RelationStore(tmp_path / "relations.sqlite")
    store.add(Relation(
        "PERSON-1", "candidate_on", "LIST-2024", country="HR", election="EP2024",
        valid_from="2024-04-01", valid_to="2024-06-09",
        source_ref="synthetic://candidate", provenance="synthetic_public_fixture",
    ))
    store.add(Relation(
        "PERSON-1", "candidate_on", "LIST-2020", country="HR", election="PARL2020",
        source_ref="synthetic://old", provenance="synthetic_public_fixture",
    ))
    rows = store.related("PERSON-1", country="HR", election="EP2024")
    assert len(rows) == 1
    assert rows[0]["target_id"] == "LIST-2024"
    assert rows[0]["valid_from"] == "2024-04-01"
    assert rows[0]["source_ref"] == "synthetic://candidate"

def test_party_member_of_list_does_not_flatten_observation(tmp_path):
    store = EP24RelationStore(tmp_path / "relations.sqlite")
    store.add(Relation(
        "PARTY-A", "member_of_list", "LIST-COALITION",
        country="HR", election="EP2024", source_ref="synthetic://membership",
        provenance="synthetic_public_fixture",
    ))
    ctx = relation_context(["PARTY-A"], store, country="HR", election="EP2024")
    assert [item["obj_id"] for item in ctx["observed"]] == ["PARTY-A"]
    assert ctx["related"][0]["target_id"] == "LIST-COALITION"
    assert ctx["related"][0]["evidence_role"] == BACKGROUND_EVIDENCE_ROLE

def test_relation_type_validation(tmp_path):
    store = EP24RelationStore(tmp_path / "relations.sqlite")
    try:
        store.add(Relation("A", "flatten_to_party", "B"))
    except ValueError as exc:
        assert "unsupported EP24 relation type" in str(exc)
    else:
        raise AssertionError("unsupported relation type must fail")
