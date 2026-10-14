"""Regression checks for evidence-grounded Step 8 relation admission."""
from types import SimpleNamespace
from step_8_roihu_social_network_analysis import _network_relation_gate

def edge(src, relation, dst, quote="Exact original statement by a named actor."):
    return SimpleNamespace(source_actor=src, target_actor=dst, relation_type=relation, evidence_quote=quote)

def test_dna_support_not_a_social_tie():
    assert not _network_relation_gate([edge("Kokoomus", "support / advocacy", "EU")])

def test_political_opposition_not_interaction():
    assert not _network_relation_gate([edge("Elsi Katainen", "opposition", "European Commission")])

def test_general_electorate_not_individual_actor():
    assert not _network_relation_gate([edge("Kokoomus", "reply", "Voters (general electorate)")])

def test_genuine_mention_is_preserved():
    assert len(_network_relation_gate([edge("Iltalehti", "mention", "Alexander Stubb")])) == 1

def test_affiliation_requires_explicit_membership_evidence():
    assert not _network_relation_gate([edge("Elsi Katainen", "affiliation", "TRAN", "Raised issues in TRAN")])
    assert len(_network_relation_gate([edge("Elsi Katainen", "affiliation", "European Parliament", "I have worked in Parliament")])) == 1
