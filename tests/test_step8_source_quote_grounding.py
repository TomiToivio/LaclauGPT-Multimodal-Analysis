"""No SNA edge without a verbatim quote from the supplied evidence packet."""
from types import SimpleNamespace

from step_8_roihu_social_network_analysis import _grounded_edge_candidates


def edge(source="Alice", target="Bob", quote="Alice replied publicly to Bob"):
    return SimpleNamespace(source_actor=source, target_actor=target, evidence_quote=quote)


def test_only_source_verbatim_quote_can_form_edge():
    evidence = "[asr_transcript]\nAlice replied publicly to Bob"
    assert len(_grounded_edge_candidates([edge()], evidence)) == 1
    assert _grounded_edge_candidates([edge(quote="Alice secretly commands Bob")], evidence) == []


def test_empty_short_or_self_edge_is_rejected():
    text = "Alice replied publicly to Bob"
    assert _grounded_edge_candidates([edge(quote="")], text) == []
    assert _grounded_edge_candidates([edge(quote="Alice")], text) == []
    assert _grounded_edge_candidates([edge(source="Alice", target="Alice")], text) == []


def test_normalized_whitespace_can_preserve_original_quote():
    text = "[asr_transcript]\nAlice   replied\nto Bob at the hearing"
    assert _grounded_edge_candidates(
        [edge(quote="Alice replied to Bob at the hearing")], text
    )
