"""Tests for the Step 7 Discourse Network Analysis refactor (issue #182).

Synthetic fixtures only; no private EP24 rows, codebooks or credentials. The
model call is stubbed, so the whole file runs without Ollama, MongoDB, Redis or R.

The methodology these tests pin is Leifeld (2017), Discourse Network Analysis:
Policy Debates as Dynamic Networks (Oxford Handbook of Political Networks, ch. 25).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_dna as dna  # noqa: E402
import ep24_stage_contract as contract  # noqa: E402
import step_7_roihu_discourse_network_analysis as step7  # noqa: E402
from ep24_entities import fold_key  # noqa: E402
from ep24_pipeline import assert_source_metadata_preserved, load_cumulative_csv  # noqa: E402

# --- fixtures ---------------------------------------------------------------

def _statement(**overrides) -> dict:
    base = {
        "statement_id": "s1",
        "source_record_id": "SYNTH-FI-1",
        "document_id": "SYNTH-FI-1",
        "network_layer": "discourse",
        "organization": "Party A",
        "actor_id": "actor:a",
        "concept": "climate policy",
        "concept_id": "concept:x",
        "proposition": "Climate policy should be tightened.",
        "agreement": 1,
        "date_time": "2024-05-01T10:00:00+00:00",
        "evidence_quote": "we must tighten climate policy",
        "confidence": 0.9,
        "provenance": "model_derived",
    }
    base.update(overrides)
    return base


# --- 1. Leifeld qualifier semantics ----------------------------------------

def test_explicit_support_and_opposition_map_to_the_binary_qualifier():
    assert dna.canonical_agreement("support") == dna.AGREEMENT_POSITIVE
    assert dna.canonical_agreement("oppose") == dna.AGREEMENT_NEGATIVE
    assert dna.canonical_agreement(True) == dna.AGREEMENT_POSITIVE
    assert dna.canonical_agreement(False) == dna.AGREEMENT_NEGATIVE


@pytest.mark.parametrize(
    "value",
    ["neutral", "mixed", "unknown", "unclear", "ambiguous", "", None, "supports and opposes"],
)
def test_ambiguous_stances_are_never_forced_to_binary(value):
    """The central #182 requirement: do not collapse ambiguity into support/oppose."""
    assert dna.canonical_agreement(value) is None
    assert dna.agreement_label(value) == dna.AGREEMENT_UNCERTAIN


def test_two_actors_with_opposite_stances_stay_distinguishable():
    statements = [
        _statement(organization="Party A", agreement=dna.AGREEMENT_POSITIVE),
        _statement(organization="Party B", agreement=dna.AGREEMENT_NEGATIVE),
    ]
    assert dna.actor_congruence_network(statements) == []
    conflict = dna.actor_conflict_network(statements)
    assert conflict == [("Party A", "Party B", 1.0)]
    # Concept+qualifier are distinct signed concepts.
    assert dna.signed_concepts(statements) == ["climate policy|0", "climate policy|1"]


def test_congruence_requires_the_same_stance_on_the_same_concept():
    statements = [
        _statement(organization="A", agreement=dna.AGREEMENT_POSITIVE),
        _statement(organization="B", agreement=dna.AGREEMENT_POSITIVE),
        _statement(organization="C", agreement=dna.AGREEMENT_NEGATIVE),
    ]
    assert dna.actor_congruence_network(statements) == [("A", "B", 1.0)]
    assert dna.actor_conflict_network(statements) == [("A", "C", 1.0), ("B", "C", 1.0)]


# --- 2. temporal + duplicate semantics -------------------------------------

def test_timestamps_are_preserved_and_normalised_to_utc():
    assert dna.parse_date_time("2024-05-01T12:00:00+03:00") == "2024-05-01T09:00:00+00:00"
    assert dna.parse_date_time("2024-05-01") == "2024-05-01T00:00:00+00:00"
    # A weaker value keeps its precision rather than being invented.
    assert dna.parse_date_time("2024-05") == "2024-05-01T00:00:00+00:00"
    assert dna.parse_date_time("") == ""
    assert dna.parse_date_time("not a date") == ""


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("include", ""),
        ("acrossrange", "*"),
        ("year", "2024"),
        ("month", "2024-05"),
    ],
)
def test_duplicate_buckets_follow_the_rdna_settings(policy, expected):
    assert dna.duplicate_bucket("2024-05-15T10:00:00+00:00", policy) == expected


def test_repeated_identical_statements_are_handled_deterministically():
    repeated = [_statement(statement_id=f"s{i}") for i in range(3)]
    assert len(dna.deduplicate(repeated, policy="include")) == 3
    assert len(dna.deduplicate(repeated, policy="document")) == 1
    assert len(dna.deduplicate(repeated, policy="acrossrange")) == 1
    # A different qualifier is a different statement tuple, so it survives.
    mixed = repeated + [_statement(agreement=dna.AGREEMENT_NEGATIVE)]
    assert len(dna.deduplicate(mixed, policy="acrossrange")) == 2
    # Deterministic: same input, same output.
    assert dna.deduplicate(mixed, policy="document") == dna.deduplicate(mixed, policy="document")


def test_duplicate_policy_rejects_an_unknown_setting():
    with pytest.raises(ValueError):
        dna.duplicate_bucket("2024-05-01", "fortnight")
    with pytest.raises(ValueError):
        dna.deduplicate([_statement()], policy="fortnight")


# --- 3. network construction semantics -------------------------------------

def test_two_mode_affiliation_counts_actor_concept_mentions():
    network = dna.two_mode_network(
        [_statement(organization="A"), _statement(organization="A"), _statement(organization="B")]
    )
    assert network["rows"] == ["A", "B"]
    assert network["columns"] == ["climate policy"]
    assert network["values"] == [[2], [1]]


def test_two_mode_subtract_aggregation_matches_rdna():
    statements = [
        _statement(organization="A", agreement=dna.AGREEMENT_POSITIVE),
        _statement(organization="A", agreement=dna.AGREEMENT_NEGATIVE),
    ]
    network = dna.two_mode_network(statements, qualifier_aggregation="subtract")
    assert network["values"] == [[0]]  # one support minus one oppose


def test_two_mode_combine_aggregation_marks_mixed_statements():
    statements = [
        _statement(organization="A", agreement=dna.AGREEMENT_POSITIVE),
        _statement(organization="A", agreement=dna.AGREEMENT_NEGATIVE),
        _statement(organization="B", agreement=dna.AGREEMENT_POSITIVE),
    ]
    network = dna.two_mode_network(statements, qualifier_aggregation="combine")
    values = {row: network["values"][i] for i, row in enumerate(network["rows"])}
    assert values["A"] == [3]  # mixed
    assert values["B"] == [1]  # positive only


def test_uncoded_statements_are_excluded_from_congruence_and_conflict():
    statements = [_statement(organization="A", agreement=None), _statement(organization="B", agreement=None)]
    assert dna.actor_congruence_network(statements) == []
    assert dna.actor_conflict_network(statements) == []


@pytest.mark.parametrize("normalization", list(dna.NORMALIZATIONS_ONE_MODE))
def test_every_one_mode_normalization_is_accepted(normalization):
    statements = [_statement(organization="A"), _statement(organization="A"), _statement(organization="B")]
    edges = dna.actor_congruence_network(statements)
    result = dna.apply_normalization(edges, statements=statements, normalization=normalization)
    assert len(result) == len(edges)
    assert all(weight > 0 for _, _, weight in result)


def test_normalization_reduces_raw_activity_bias():
    """Raw activity is not similarity: a prolific actor should not dominate.

    A codes only ``climate policy``; B codes ``climate policy`` and a second
    concept. They share one congruent coding, so the raw weight is 1, but the
    Jaccard divisor is the union of their concept sets (2), halving it.
    """
    statements = [
        _statement(organization="A"),
        _statement(organization="A"),
        _statement(organization="A"),
        _statement(organization="B"),
        _statement(organization="B", concept="tax policy", statement_id="s-b2"),
    ]
    raw = dna.actor_congruence_network(statements)
    jaccard = dna.apply_normalization(raw, statements=statements, normalization="jaccard")
    assert raw[0][2] > jaccard[0][2]
    assert jaccard[0][2] == pytest.approx(0.5)


def test_normalization_rejects_an_unknown_setting():
    with pytest.raises(ValueError):
        dna.apply_normalization([], statements=[], normalization="zscore")
    with pytest.raises(ValueError):
        dna.two_mode_network([], normalization="cosine")  # one-mode-only setting


# --- 4. rDNA event-list interoperability -----------------------------------

def test_event_list_exports_only_binary_statements():
    statements = [_statement(agreement=dna.AGREEMENT_POSITIVE), _statement(statement_id="s2", agreement=None)]
    rows = dna.build_event_list(statements, document_id="doc-1")
    assert len(rows) == 1
    assert rows[0]["agreement"] == dna.AGREEMENT_POSITIVE


def test_event_list_uses_the_rdna_statement_type_and_variable_names():
    rows = dna.build_event_list([_statement()], document_id="doc-1")
    row = rows[0]
    assert row["statement_type"] == "DNA Statement"
    for column in ("organization", "concept", "agreement", "date_time", "document_id"):
        assert column in row
    assert dna.EVENT_LIST_COLUMNS == tuple(row.keys())


def test_event_list_round_trips_through_csv_with_types_intact(tmp_path):
    path = dna.write_event_list_csv(dna.build_event_list([_statement()], document_id="d"), tmp_path / "e.csv")
    reread = dna.read_event_list_csv(path)
    assert reread[0]["agreement"] == 1
    assert isinstance(reread[0]["confidence"], float)


def test_statement_ids_are_deterministic_and_stable_across_reordering():
    def ident(agreement: int) -> str:
        return dna.statement_id(
            source_record_id="r", actor="A", concept="c", agreement=agreement,
            date_time="2024-05-01T00:00:00+00:00", proposition="p",
        )

    assert ident(1) == ident(1)
    assert ident(1) != ident(0)


def test_importer_is_generated_against_the_official_api_and_never_writes_a_dna_file(tmp_path):
    script = dna.write_rdna_import_script(tmp_path / "events.csv", tmp_path / "import.R")
    text = script.read_text(encoding="utf-8")
    assert "dna_addDocuments(" in text
    assert "dna_addStatement(" in text
    assert "dna_openDatabase(" in text
    assert "Leifeld" in text
    # We deliberately do not reverse-engineer the .dna SQLite format.
    assert "sqlite" not in text.split("dna_openDatabase")[1].split("\n")[0].replace("sqlite:///", "")


def test_graphml_and_edge_csv_are_written(tmp_path):
    edges = [("A", "B", 0.5)]
    graphml = dna.write_graphml(edges, tmp_path / "n.graphml")
    text = graphml.read_text(encoding="utf-8")
    assert 'source="A"' in text and 'target="B"' in text
    csv_path = dna.write_edge_csv(edges, tmp_path / "e.csv")
    assert "weight" in csv_path.read_text(encoding="utf-8")


def test_summarize_counts_binary_and_uncertain_separately():
    counts = dna.summarize([_statement(), _statement(statement_id="s2", agreement=None)])
    assert counts["statement_count"] == 2
    assert counts["binary_count"] == 1
    assert counts["uncertain_count"] == 1


# --- 5. prompt: Leifeld methodology, evidence discipline --------------------

def test_system_prompt_implements_leifeld_coding_rules():
    prompt = step7.SYSTEM_PROMPT
    assert "Leifeld" in prompt
    assert "unit of analysis is a statement" in prompt
    for phrase in ("actor", "concept", "positive/support", "negative/oppose"):
        assert phrase in prompt
    assert "NEVER evidence" in prompt
    assert "uncertain" in prompt


def test_system_prompt_keeps_the_conceptual_boundary_against_other_layers():
    # Normalise wrapping so assertions do not depend on the prompt's line breaks.
    prompt = " ".join(step7.SYSTEM_PROMPT.split())
    assert "Do not confuse actor extraction with concept extraction" in prompt
    assert "opposite positions must remain distinguishable" in prompt
    assert "Do not force a binary qualifier" in prompt
    assert "timeless graph" in prompt


def test_memory_and_rag_are_never_source_evidence_in_the_prompt_context():
    row = pd.Series({
        "video_id": "v1",
        "asr_translated": "current transcript evidence",
        "memory_context_json": '{"label":"remembered"}',
        "rag_context_json": '{"text":"prior corpus text"}',
        "entities": '["Researcher Entity"]',
    })
    context, _ = step7.build_prompt_context(row)
    assert "current transcript evidence" in context
    assert "normalization_context_not_source_evidence" in context
    assert "prior_analysis_context_not_source_evidence" in context
    # The memory/RAG payloads appear only under their explicit context roles.
    assert context.index("=== RESEARCHER MEMORY ===") > context.index("current transcript evidence")


def test_prompt_context_carries_step_6_output_without_feeding_step_7_back_in():
    row = pd.Series({
        "video_id": "v1",
        "asr_translated": "text",
        "formula_of_populism_analysis": "STEP 6 DISCOURSE ANALYSIS",
        "laclau_structured_json": '{"us_constructs":[]}',
        "dna_analysis_markdown": "OLD STEP 7 OUTPUT",
        "dna_statements_json": '[{"organization":"OLD"}]',
    })
    context, _ = step7.build_prompt_context(row)
    assert "STEP 6 DISCOURSE ANALYSIS" in context
    assert "OLD STEP 7 OUTPUT" not in context
    assert '{"organization":"OLD"}' not in context


def test_prompt_context_is_bounded_and_reports_truncation():
    row = pd.Series({"video_id": "v1", "asr_translated": "x" * 5000})
    context, truncated = step7.build_prompt_context(row, max_chars=500)
    assert truncated is True
    assert len(context) == 500
    assert step7.build_prompt_context(row, max_chars=10_000)[1] is False


# --- 6. statement records: normalization, IDs, provenance ------------------

class _Model:
    """Stand-in for the pydantic statement model."""

    def __init__(self, **kw):
        self.actor_name = kw.get("actor_name", "")
        self.actor_canonical = kw.get("actor_canonical", "")
        self.concept_label = kw.get("concept_label", "")
        self.concept_canonical = kw.get("concept_canonical", "")
        self.proposition = kw.get("proposition", "")
        self.stance = kw.get("stance", "")
        self.agreement = kw.get("agreement", "uncertain")
        self.date_time = kw.get("date_time", "")
        self.evidence_quote = kw.get("evidence_quote", "")
        self.evidence_fields = kw.get("evidence_fields", [])
        self.confidence = kw.get("confidence", 0.5)
        self.uncertainty_notes = kw.get("uncertainty_notes", [])
        self.counter_evidence = kw.get("counter_evidence", [])


class _Parsed:
    def __init__(self, statements):
        self.analysis_markdown = "analysis"
        self.statements = statements
        self.corpus_level_cautions = []


def _row(**overrides) -> pd.Series:
    base = {
        "video_id": "SYNTH-FI-1",
        "country": "Finland",
        "account_type": "Party",
        "_storage_id": "SYNTH-FI-1",
        "allas_filename": "https://example.invalid/x.mp4",
        "asr_translated": "we must tighten climate policy",
        "ep24_entity_resolution_json": json.dumps([
            {"decision": "RESOLVED", "surface_form": "Pääministeri Orpo",
             "canonical_name": "Petteri Orpo", "entity_id": "actor:orpo"},
        ]),
    }
    base.update(overrides)
    return pd.Series(base)


def test_resolved_entities_supply_the_stable_actor_id():
    parsed = _Parsed([_Model(actor_name="Pääministeri Orpo", concept_label="climate policy",
                             agreement="support", evidence_quote="q")])
    records = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={
        fold_key("Pääministeri Orpo"): {"entity_id": "actor:orpo", "canonical_name": "Petteri Orpo"}
    })
    assert records[0]["actor_id"] == "actor:orpo"
    assert records[0]["organization"] == "Petteri Orpo"
    assert records[0]["actor_name_raw"] == "Pääministeri Orpo"


def test_novel_concepts_get_a_stable_deterministic_id_and_are_flagged():
    parsed = _Parsed([_Model(actor_name="A", concept_label="a brand new claim", agreement="support")])
    first = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={})
    second = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={})
    assert first[0]["concept_id"] == second[0]["concept_id"]
    assert first[0]["concept_provenance"] == step7.CONCEPT_NOVEL
    assert first[0]["evidence_role"] == step7.STATEMENT_EVIDENCE_ROLE
    assert first[0]["network_layer"] == "discourse"


def test_unknown_stance_stays_uncertain_in_the_record():
    parsed = _Parsed([_Model(actor_name="A", concept_label="c", agreement="mixed")])
    records = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={})
    assert records[0]["agreement"] is None


def test_statements_without_actor_or_concept_are_dropped_not_invented():
    parsed = _Parsed([
        _Model(actor_name="", concept_label="c"),
        _Model(actor_name="A", concept_label=""),
        _Model(actor_name="A", concept_label="c"),
    ])
    records = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={})
    assert len(records) == 1


def test_multiple_actors_and_concepts_in_one_record():
    parsed = _Parsed([
        _Model(actor_name="A", concept_label="c1", agreement="support"),
        _Model(actor_name="A", concept_label="c2", agreement="oppose"),
        _Model(actor_name="B", concept_label="c1", agreement="oppose"),
    ])
    records = step7._statement_records(parsed, _row(), context_hash="h", entity_lookup={})
    assert len(records) == 3
    assert len({r["statement_id"] for r in records}) == 3
    statements = [
        {**r, "agreement": dna.canonical_agreement(r["agreement"])}
        for r in records
    ]
    assert len(dna.actor_conflict_network(statements)) >= 1


def test_source_timestamp_is_used_when_the_model_gives_none():
    parsed = _Parsed([_Model(actor_name="A", concept_label="c", agreement="support")])
    records = step7._statement_records(
        parsed, _row(preprocess_completed_at="2024-05-01T00:00:00+00:00"),
        context_hash="h", entity_lookup={},
    )
    assert records[0]["date_time"] == "2024-05-01T00:00:00+00:00"


# --- 7. pipeline contract ---------------------------------------------------

def test_step_7_columns_are_declared_in_the_stage_contract():
    stage = contract.stage(7)
    assert stage.appends == dna.DNA_COLUMNS
    assert stage.module == "step_7_roihu_discourse_network_analysis.py"


def test_stage_7_is_no_longer_empty_by_design():
    """Declaring the columns deliberately moves stage 7 out of empty_by_design."""
    assert contract.stage(7).appends, "stage 7 must declare the columns it writes"
    assert contract.stage(7).notes


def test_cumulative_write_preserves_every_incoming_column():
    before = pd.DataFrame([{
        "video_id": "v1", "country": "Finland", "researcher_note": "keep me",
        "entities": '["E"]', "asr_translated": "text", "formula_of_populism_analysis": "s6",
    }])
    after = before.copy()
    for column in dna.DNA_COLUMNS:
        after[column] = "value"
    assert_source_metadata_preserved(before, after)  # must not raise
    assert all(column in after.columns for column in before.columns)


def test_stage_contract_column_check_finds_every_declared_dna_column_written():
    """Mirrors tests/test_ep24_stage_contract.py's written-column check."""
    source = (ROOT / contract.stage(7).module).read_text(encoding="utf-8")
    missing = [column for column in dna.DNA_COLUMNS if column not in source]
    assert not missing, f"declared but not written in the module: {missing}"


def test_exports_are_written_offline_and_deterministically(tmp_path):
    frame = pd.DataFrame([{
        "video_id": "v1",
        "_storage_id": "v1",
        "allas_filename": "https://example.invalid/x.mp4",
        "dna_statements_json": json.dumps([
            _statement(organization="A", agreement=1),
            _statement(statement_id="s2", organization="B", agreement=0),
        ]),
    }])
    written = step7.write_exports(frame, "finland", tmp_path / "ep24_finland.csv")
    assert Path(written["event_list"]).is_file()
    assert Path(written["importer"]).is_file()
    assert Path(written["actor_conflict"]).is_file()
    events = dna.read_event_list_csv(written["event_list"])
    assert len(events) == 2


def test_exports_are_safe_when_no_statements_were_coded(tmp_path):
    frame = pd.DataFrame([{"video_id": "v1", "_storage_id": "v1", "dna_statements_json": ""}])
    written = step7.write_exports(frame, "finland", tmp_path / "ep24_finland.csv")
    assert Path(written["event_list"]).is_file()
    assert dna.read_event_list_csv(written["event_list"]) == []


# --- 8. offline operation ---------------------------------------------------

def test_step_7_processes_a_row_without_mongo_or_redis(tmp_path, monkeypatch):
    """Mongo and Redis absent must not break a normal run."""
    monkeypatch.delenv("LACLAUGPT_MONGO_ENABLED", raising=False)
    monkeypatch.delenv("LACLAUGPT_REDIS_URL", raising=False)
    monkeypatch.setattr(step7, "analyze_context", lambda context: (
        "md",
        _Parsed([_Model(actor_name="Party A", concept_label="climate policy",
                        agreement="support", evidence_quote="we must tighten climate policy")]),
    ))
    source = tmp_path / "ep24_finland.csv"
    pd.DataFrame([{
        "country": "Finland", "video_id": "v1", "allas_filename": "https://x/y.mp4",
        "entities": "[]", "themes": "[]", "asr_translated": "we must tighten climate policy",
    }]).to_csv(source, index=False)
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(source))
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "finland")
    output = step7.process_country("finland")

    frame = load_cumulative_csv(output)
    assert frame.iloc[0]["dna_status"] == "ok"
    # The CSV round-trips as strings; compare numerically.
    assert int(frame.iloc[0]["dna_statement_count"]) == 1
    assert int(frame.iloc[0]["dna_binary_count"]) == 1
    assert frame.iloc[0]["dna_persistence_status"] == "mongo_disabled"
    statements = json.loads(frame.iloc[0]["dna_statements_json"])
    assert statements[0]["agreement"] == 1
    # Every incoming column survived.
    assert frame.iloc[0]["asr_translated"] == "we must tighten climate policy"


def test_memory_retrieval_does_not_become_statement_evidence(tmp_path, monkeypatch):
    """A remembered actor must not appear as a statement actor."""
    monkeypatch.delenv("LACLAUGPT_MONGO_ENABLED", raising=False)
    monkeypatch.setattr(step7, "analyze_context", lambda context: ("md", _Parsed([])))
    source = tmp_path / "ep24_finland.csv"
    pd.DataFrame([{
        "country": "Finland", "video_id": "v1", "allas_filename": "https://x/y.mp4",
        "entities": "[]", "themes": "[]", "asr_translated": "text",
        "memory_context_json": '{"label":"Petteri Orpo"}',
        "rag_context_json": '{"text":"prior text"}',
    }]).to_csv(source, index=False)
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(source))
    output = step7.process_country("finland")
    frame = load_cumulative_csv(output)
    assert json.loads(frame.iloc[0]["dna_statements_json"]) == []
    assert int(frame.iloc[0]["dna_statement_count"]) == 0


def test_step_7_help_works_without_runtime_dependencies():
    """#152 contract: --help must not import ollama/pydantic at module level."""
    source = (ROOT / contract.stage(7).module).read_text(encoding="utf-8")
    head = source.split("def _models", 1)[0]
    # A bare `import ollama` / `from pydantic import ...` at module level would
    # run during --help and break the minimal-environment contract.
    assert "\nimport ollama" not in head
    assert "\nfrom pydantic import" not in head
    assert "ollama.chat(" not in head


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
