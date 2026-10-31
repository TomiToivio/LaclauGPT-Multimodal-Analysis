"""Merge proposals for private EP24 codebooks (issue #333): deterministic, read-only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "propose_codebook_merges",
    Path(__file__).resolve().parents[1] / "scripts" / "ep24" / "propose_codebook_merges.py",
)
tool = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(tool)


def codebook(**entries):
    return {"entries": entries["entries"]}


class TestNormalise:
    def test_case_whitespace_and_punctuation_fold(self):
        assert tool.normalise("  #Perus   suomalaiset ") == "perus suomalaiset"
        assert tool.normalise("Civic Platform (PO)") == "civic platform po"


class TestProposals:
    def test_case_variants_are_proposed(self):
        document = codebook(entries=[
            {"kind": "theme", "label": "Same Theme", "aliases": []},
            {"kind": "theme", "label": "same theme", "aliases": []},
        ])
        found = tool.proposals(document)
        assert len(found) == 1
        assert found[0]["reason"] == "same_normalised_label"

    def test_word_order_is_not_guessed(self):
        document = codebook(entries=[
            {"kind": "theme", "label": "EU-Poland relations", "aliases": []},
            {"kind": "theme", "label": "Poland-EU relations", "aliases": []},
        ])
        assert tool.proposals(document) == []

    def test_shared_alias_groups_entries(self):
        document = codebook(entries=[
            {"kind": "entity", "label": "Party A", "aliases": ["Party"]},
            {"kind": "entity", "label": "Party B", "aliases": ["Party"]},
        ])
        found = tool.proposals(document)
        assert len(found) == 1
        assert found[0]["reason"] == "shared_alias"
        assert found[0]["entries"] == ["Party A", "Party B"]

    def test_distinct_kinds_are_not_merged(self):
        document = codebook(entries=[
            {"kind": "entity", "label": "Same", "aliases": []},
            {"kind": "theme", "label": "Same", "aliases": []},
        ])
        assert tool.proposals(document) == []

    def test_empty_aliases_do_not_create_groups(self):
        document = codebook(entries=[
            {"kind": "entity", "label": "A", "aliases": ["", "  "]},
            {"kind": "entity", "label": "B", "aliases": ["", "  "]},
        ])
        assert tool.proposals(document) == []

    def test_duplicate_labels_in_one_entry_are_not_a_merge(self):
        document = codebook(entries=[
            {"kind": "entity", "label": "A", "aliases": ["A", "A"]},
        ])
        assert tool.proposals(document) == []


class TestCli:
    def test_missing_file_exits_two(self, tmp_path: Path):
        assert tool.main(["--codebook", str(tmp_path / "nope.json")]) == 2

    def test_json_output_counts_proposals(self, tmp_path: Path, capsys):
        path = tmp_path / "cb.json"
        path.write_text(json.dumps(codebook(entries=[
            {"kind": "entity", "label": "A", "aliases": ["X"]},
            {"kind": "entity", "label": "B", "aliases": ["X"]},
        ])))
        assert tool.main(["--codebook", str(path), "--json"]) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["count"] == 1
