from roihu_memory import EP24Memory


def _obj(memory, label, entity_type):
    return memory.add_object(
        "actor",
        label,
        country="HR",
        language="hr",
        entity_type=entity_type,
        state="CANONICAL",
        origin="synthetic-test",
        locked=True,
    )


def test_electoral_list_relations_do_not_flatten_observed_evidence(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")

    standalone = _obj(memory, "Independent List", "electoral_list")
    coalition = _obj(memory, "Forward Together List", "electoral_list")
    party_a = _obj(memory, "Civic Party", "party")
    party_b = _obj(memory, "Green Party", "party")
    candidate = _obj(memory, "Ana Example", "person")

    memory.add_alias(party_a, "CP", country="HR", language="hr", provenance="synthetic")
    memory.add_relation(
        coalition, "has_member", party_a,
        country="HR", election="EP2024", valid_from="2024-04-01", valid_to="2024-06-09",
        source_type="official-election-fixture", source_ref="synthetic://hr/ep2024/list",
        publication_date="2024-04-01", evidence_locator="list:1",
    )
    memory.add_relation(
        party_a, "member_of_list", coalition,
        country="HR", election="EP2024", valid_from="2024-04-01", valid_to="2024-06-09",
        source_type="official-election-fixture", source_ref="synthetic://hr/ep2024/list",
        publication_date="2024-04-01", evidence_locator="party:cp",
    )
    memory.add_relation(
        coalition, "has_member", party_b,
        country="HR", election="EP2024", valid_from="2024-04-01", valid_to="2024-06-09",
        source_type="official-election-fixture", source_ref="synthetic://hr/ep2024/list",
        publication_date="2024-04-01", evidence_locator="party:green",
    )
    memory.add_relation(
        candidate, "candidate_on", coalition,
        country="HR", election="EP2024", valid_from="2024-04-01", valid_to="2024-06-09",
        source_type="official-election-fixture", source_ref="synthetic://hr/ep2024/candidates",
        publication_date="2024-04-01", evidence_locator="candidate:ana",
    )

    result = memory.resolve_with_relations(
        "Forward Together List", "actor",
        country="HR", language="hr", election="EP2024", at_date="2024-05-15",
    )

    assert result["decision"] == "EXISTING"
    assert result["obj_id"] == coalition
    assert result["observed_in_current_item"] is True
    assert {r["object_id"] for r in result["related_context"] if r["subject_id"] == coalition} == {party_a, party_b}
    assert all(r["observed_in_current_item"] is False for r in result["related_context"])
    assert "must never create actor/entity evidence" in result["prompt_firewall"]

    # The standalone list remains a distinct list object, not a party surrogate.
    standalone_result = memory.resolve_with_relations(
        "Independent List", "actor", country="HR", language="hr", election="EP2024"
    )
    assert standalone_result["obj_id"] == standalone
    assert standalone_result["related_context"] == []


def test_relation_scope_time_and_provenance_are_preserved(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    coalition = _obj(memory, "Forward Together List", "electoral_list")
    party = _obj(memory, "Civic Party", "party")

    memory.add_relation(
        coalition, "has_member", party,
        country="HR", election="EP2024", valid_from="2024-04-01", valid_to="2024-06-09",
        source_type="official-election-fixture", source_ref="synthetic://hr/ep2024/list",
        source_language="hr", publication_date="2024-04-01", evidence_locator="row:7",
    )

    active = memory.related_context(coalition, country="HR", election="EP2024", at_date="2024-05-01")
    assert len(active) == 1
    rel = active[0]
    assert rel["country"] == "HR"
    assert rel["election"] == "EP2024"
    assert rel["valid_from"] == "2024-04-01"
    assert rel["valid_to"] == "2024-06-09"
    assert rel["source_ref"] == "synthetic://hr/ep2024/list"
    assert rel["source_language"] == "hr"
    assert rel["evidence_locator"] == "row:7"
    assert rel["evidence_role"] == "background_relation_not_observed_evidence"

    assert memory.related_context(coalition, country="HR", election="EP2024", at_date="2024-07-01") == []
    assert memory.related_context(coalition, country="HR", election="NATIONAL2024") == []


def test_ambiguous_short_abbreviation_still_abstains_before_relations(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    p1 = _obj(memory, "Civic Party", "party")
    p2 = _obj(memory, "Citizens Platform", "party")
    memory.add_alias(p1, "CP", country="HR", language="hr", provenance="synthetic")
    memory.add_alias(p2, "CP", country="HR", language="hr", provenance="synthetic")

    result = memory.resolve_with_relations("CP", "actor", country="HR", language="hr", election="EP2024")
    assert result["decision"] == "AMBIGUOUS"
    assert result["obj_id"] == ""
    assert result["observed_in_current_item"] is False
    assert result["related_context"] == []


def test_relations_are_exported_for_csv_pandas_compatibility(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    coalition = _obj(memory, "Forward Together List", "electoral_list")
    party = _obj(memory, "Civic Party", "party")
    memory.add_relation(coalition, "has_member", party, country="HR", election="EP2024", source_ref="synthetic://x")

    outputs = memory.export_csv(tmp_path / "csv")
    names = {path.name for path in outputs}
    assert "memory_relations.csv" in names
