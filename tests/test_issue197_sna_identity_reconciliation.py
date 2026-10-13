"""Regression tests for issue #197: SNA node identity must not fragment one actor.

The defect: ``ep24_sna.node_identity`` prefers the canonical entity id, but Step 8's
enrichment only supplies one when the mention matches a *resolved label exactly*
(``resolution_lookup`` keys on ``surface_form`` / ``normalized_form`` /
``canonical_name``, each folded). A bare surname, a title-prefixed form or an
inflected form therefore became its own ``actor:<hash>`` node **even though the row
contained a RESOLVED record for that person**, splitting one actor and inflating
``node_count``, ``density``, ``degree`` and ``component_count``.

The policy under test is the one specified in the issue: merge **only when exactly
one** resolved candidate in the row matches, by exact-folded, title-stripped or
one-directional token-subset equality. Zero or ≥2 candidates ⇒ no merge, and the
node stays separate and is reported.

Rationale carried from the #196 review, as the issue asks:

* pairing ``ep24_entity_ids`` with mentions by list position produced three nodes
  for one actor, because the id list is deduplicated while the mention list is not;
* rows carrying registry records produced ``entity:<id>`` while rows without them
  produced ``actor:<hash>`` for the same person.

Both are the same failure mode: identity derived from anything other than a
canonical decision.

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""
from __future__ import annotations

import json
import unittest

from ep24_sna import (
    MIN_SUBSET_TOKEN_LENGTH,
    apply_reconciliation_metrics,
    basic_metrics,
    graph_from_edges,
    identity_fragmentation,
    node_identity,
    reconcile_actor_nodes,
)


def resolution_json(*records) -> str:
    return json.dumps(list(records))


def record(surface, canonical, entity_id, normalized=None, decision="RESOLVED"):
    return {
        "surface_form": surface,
        "normalized_form": normalized or surface,
        "canonical_name": canonical,
        "entity_id": entity_id,
        "decision": decision,
    }


def row_with(*records, **extra):
    row = {
        "video_id": "ITEM-1",
        "country": "Finland",
        "source_type": "TikTok",
        "ep24_entity_resolution_json": resolution_json(*records),
    }
    row.update(extra)
    return row


def edge_of(source_label, target_label, source_id="", target_id=""):
    return {
        "source_actor": source_label,
        "target_actor": target_label,
        "source_actor_id": source_id,
        "target_actor_id": target_id,
        "relation_type": "mentions",
        "evidence_quote": "q",
        "confidence": 0.9,
    }


ORPO = record("Petteri Orpo", "Petteri Orpo", "FI-ORPO")


class TheDefectItselfTests(unittest.TestCase):
    """The reproduction from the issue, so the fix is anchored to the real failure."""

    def test_bare_surname_hashes_to_a_different_node(self) -> None:
        self.assertNotEqual(node_identity("Petteri Orpo"), node_identity("Orpo"))

    def test_graph_path_splits_one_actor_into_two_nodes(self) -> None:
        nodes, _ = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row_with(ORPO))
        self.assertEqual(len(nodes), 2, "the split should be reproducible before reconciliation")


class UnambiguousMergeTests(unittest.TestCase):
    """Exactly one candidate ⇒ merge, and the merge is recorded."""

    def test_token_subset_surname_merges(self) -> None:
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row_with(ORPO)
        )
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row_with(ORPO))
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["node_id"].startswith("entity:"))
        self.assertEqual(report["merged_count"], 1)
        self.assertEqual(report["merged"][0]["merge_reason"], "token_subset")

    def test_title_prefixed_mention_merges(self) -> None:
        """A title-prefixed mention resolves to the person.

        The reason recorded is ``token_subset`` rather than ``title_stripped``
        because ``strip_titles("Pääministeri Orpo")`` is ``"Orpo"``, which is a
        subset of ``"Petteri Orpo"`` rather than equal to it. Both are legitimate
        matches; only the recorded reason differs.
        """
        nodes, edges = graph_from_edges(
            [edge_of("Pääministeri Orpo", "Petteri Orpo", "", "FI-ORPO")], row_with(ORPO)
        )
        merged, _, report = reconcile_actor_nodes(nodes, edges, row_with(ORPO))
        self.assertEqual(len(merged), 1)
        self.assertEqual(report["merged"][0]["merge_reason"], "token_subset")

    def test_a_mention_equal_to_the_bare_canonical_is_title_stripped(self) -> None:
        """The title_stripped reason fires when the stripped forms are equal."""
        row = row_with(record("Orpo", "Orpo", "FI-BARE"))
        nodes, edges = graph_from_edges(
            [edge_of("Orpo", "Pääministeri Orpo", "FI-BARE")], row
        )
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(merged), 1)
        self.assertEqual(report["merged"][0]["merge_reason"], "title_stripped")

    def test_metrics_are_corrected_by_the_merge(self) -> None:
        """The point of the fix: counts must not be inflated by the split."""
        row = row_with(ORPO)
        raw_nodes, raw_edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        before = basic_metrics(raw_nodes, raw_edges)
        nodes, edges, report = reconcile_actor_nodes(raw_nodes, raw_edges, row)
        after = apply_reconciliation_metrics(basic_metrics(nodes, edges), report)
        self.assertEqual(before["node_count"], 2)
        self.assertEqual(after["node_count"], 1)
        self.assertEqual(after["component_count"], 1)
        self.assertEqual(after["reconciled_actor_count"], 1)
        self.assertEqual(after["unresolved_actor_count"], 0)

    def test_the_edge_is_rewritten_not_dropped(self) -> None:
        """A merge must never lose evidence."""
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        _, merged_edges, _ = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(merged_edges), len(edges))
        self.assertTrue(all(e["source"].startswith("entity:") for e in merged_edges))
        self.assertTrue(all(e["target"].startswith("entity:") for e in merged_edges))

    def test_the_merge_records_what_it_absorbed_and_why(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        merged, _, _ = reconcile_actor_nodes(nodes, edges, row)
        survivor = merged[0]
        self.assertIn("actor:", survivor["merged_from"][0])
        self.assertIn("token_subset", survivor["merge_reason"])

    def test_the_rewritten_edge_records_its_original_endpoints(self) -> None:
        """Reversibility: the pre-merge identity is recoverable from the edge."""
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        _, merged_edges, _ = reconcile_actor_nodes(nodes, edges, row)
        reconciled = merged_edges[0]["provenance"].get("reconciled_from")
        self.assertIsNotNone(reconciled)
        self.assertTrue(reconciled["target"].startswith("actor:"))

    def test_three_mention_forms_collapse_to_one_actor(self) -> None:
        """The #196 list-position bug produced three nodes for one actor."""
        row = row_with(ORPO)
        raw = [
            edge_of("Petteri Orpo", "Orpo", "FI-ORPO"),
            edge_of("Pääministeri Orpo", "Petteri Orpo", "", "FI-ORPO"),
        ]
        nodes, edges = graph_from_edges(raw, row)
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(merged), 1)
        self.assertEqual(report["unresolved_count"], 0)


class AmbiguityIsNeverMergedTests(unittest.TestCase):
    """Two candidates ⇒ no merge, and the ambiguity is counted."""

    def test_two_people_sharing_a_surname_merge_nothing(self) -> None:
        row = row_with(
            record("Marta Kowalska", "Marta Kowalska", "PL-K1"),
            record("Piotr Kowalska", "Piotr Kowalska", "PL-K2"),
        )
        nodes, edges = graph_from_edges(
            [edge_of("Marta Kowalska", "Kowalska", "PL-K1")], row
        )
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        # the bare surname may match both -> ambiguous, so it stays separate
        self.assertEqual(report["ambiguous_count"], 1)
        self.assertEqual(report["merged_count"], 0)
        self.assertTrue(any(not n["node_id"].startswith("entity:") for n in merged))

    def test_the_ambiguous_node_is_reported_for_review(self) -> None:
        row = row_with(
            record("Marta Kowalska", "Marta Kowalska", "PL-K1"),
            record("Piotr Kowalska", "Piotr Kowalska", "PL-K2"),
        )
        nodes, edges = graph_from_edges([edge_of("Marta Kowalska", "Kowalska", "PL-K1")], row)
        _, _, report = reconcile_actor_nodes(nodes, edges, row)
        entry = report["ambiguous"][0]
        self.assertEqual(entry["label"], "Kowalska")
        self.assertEqual(sorted(entry["candidates"]), ["entity:PL-K1", "entity:PL-K2"])

    def test_ambiguous_count_reaches_the_metrics(self) -> None:
        row = row_with(
            record("Marta Kowalska", "Marta Kowalska", "PL-K1"),
            record("Piotr Kowalska", "Piotr Kowalska", "PL-K2"),
        )
        nodes, edges = graph_from_edges([edge_of("Marta Kowalska", "Kowalska", "PL-K1")], row)
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        metrics = apply_reconciliation_metrics(basic_metrics(merged, merged_edges), report)
        self.assertEqual(metrics["ambiguous_actor_count"], 1)
        self.assertTrue(metrics["ambiguous_actors"])


class NoCandidateStaysSeparateTests(unittest.TestCase):
    """No match ⇒ the node stays exactly as it was."""

    def test_an_unrelated_name_is_not_merged(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Completely Different", "FI-ORPO")], row
        )
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(report["merged_count"], 0)
        self.assertEqual(report["unresolved_count"], 1)
        self.assertEqual(len(merged), 2)

    def test_a_row_with_no_resolution_records_changes_nothing(self) -> None:
        row = row_with()
        nodes, edges = graph_from_edges([edge_of("A", "B")], row)
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(merged), len(nodes))
        self.assertEqual(len(merged_edges), len(edges))
        self.assertEqual(report["merged_count"], 0)
        self.assertEqual(report["unresolved_count"], 2)

    def test_unresolved_actors_are_counted_in_the_metrics(self) -> None:
        row = row_with()
        nodes, edges = graph_from_edges([edge_of("A", "B")], row)
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        metrics = apply_reconciliation_metrics(basic_metrics(merged, merged_edges), report)
        self.assertEqual(metrics["unresolved_actor_count"], 2)
        fragmentation = identity_fragmentation(merged)
        self.assertEqual(fragmentation["unresolved_actor_count"], 2)


class ConservativeGuardTests(unittest.TestCase):
    """The over-reach guards. Each of these would be a false merge."""

    def test_a_short_token_does_not_subset_match(self) -> None:
        """A real 1-2 character token must not subset-match a longer name.

        The guard needs a token that IS a verbatim candidate token but too short —
        "Le" in "Le Pen", "Li" in "Li Andersson". (A casefolded prefix like "An" for
        "Anna" fails the subset test anyway, so it does not exercise the guard.)
        """
        for mention, canonical in (("Le", "Le Pen"), ("Li", "Li Andersson"), ("X", "X Y Z")):
            with self.subTest(mention=mention):
                row = row_with(record(canonical, canonical, "E-1"))
                nodes, edges = graph_from_edges([edge_of(canonical, mention, "E-1")], row)
                _, _, report = reconcile_actor_nodes(nodes, edges, row)
                self.assertEqual(
                    report["merged_count"], 0,
                    f"the short token {mention!r} was merged into {canonical!r}",
                )

    def test_the_short_token_guard_is_load_bearing(self) -> None:
        """With the guard relaxed to 1, those same tokens DO merge — so it bites."""
        row = row_with(record("Le Pen", "Le Pen", "E-1"))
        nodes, edges = graph_from_edges([edge_of("Le Pen", "Le", "E-1")], row)
        _, _, report = reconcile_actor_nodes(nodes, edges, row, min_token_length=1)
        self.assertEqual(report["merged_count"], 1)

    def test_the_minimum_token_length_is_configurable(self) -> None:
        row = row_with(record("Anna Something", "Anna Something", "FI-ANNA"))
        nodes, edges = graph_from_edges([edge_of("Anna Something", "Anna", "FI-ANNA")], row)
        _, _, report = reconcile_actor_nodes(nodes, edges, row, min_token_length=2)
        self.assertEqual(report["merged_count"], 1)
        self.assertGreaterEqual(MIN_SUBSET_TOKEN_LENGTH, 3)

    def test_a_superset_mention_is_not_merged(self) -> None:
        """Direction matters: 'Petteri Orpo Something' must not merge into 'Petteri Orpo'."""
        row = row_with(ORPO)
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Petteri Orpo Extra", "FI-ORPO")], row
        )
        _, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(report["merged_count"], 0)

    def test_two_different_people_are_not_merged_by_a_shared_token(self) -> None:
        """'Petteri Orpo' and 'Petteri Virtanen' share a given name only."""
        row = row_with(
            record("Petteri Orpo", "Petteri Orpo", "FI-ORPO"),
            record("Petteri Virtanen", "Petteri Virtanen", "FI-VIRT"),
        )
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Petteri Virtanen", "FI-ORPO", "FI-VIRT")], row
        )
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(report["merged_count"], 0)
        self.assertEqual(len(merged), 2)

    def test_a_mention_matching_no_candidate_but_sharing_a_token_is_not_merged(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Orpo Something Else", "X", "FI-ORPO")], row)
        _, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(report["merged_count"], 0)


class IntegrityTests(unittest.TestCase):
    """Structural guarantees: the graph must stay closed and deterministic."""

    def test_every_edge_endpoint_exists_after_reconciliation(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Orpo", "FI-ORPO"), edge_of("Orpo", "Other")], row
        )
        merged, merged_edges, _ = reconcile_actor_nodes(nodes, edges, row)
        ids = {n["node_id"] for n in merged}
        for edge in merged_edges:
            self.assertIn(edge["source"], ids)
            self.assertIn(edge["target"], ids)

    def test_reconciliation_is_deterministic(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        first = reconcile_actor_nodes(nodes, edges, row)
        second = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual([n["node_id"] for n in first[0]], [n["node_id"] for n in second[0]])
        self.assertEqual(first[2]["merged"], second[2]["merged"])

    def test_it_does_not_mutate_its_inputs(self) -> None:
        """The caller's node and edge dicts must be untouched, deeply.

        Checking only the top-level ids was too weak: sharing the inner dicts let a
        mutation that aliased the caller's list still pass.
        """
        import copy

        row = row_with(ORPO)
        nodes, edges = graph_from_edges(
            [edge_of("Petteri Orpo", "Orpo", "FI-ORPO"),
             edge_of("Orpo", "Other", "FI-ORPO")], row
        )
        nodes_snapshot = copy.deepcopy(nodes)
        edges_snapshot = copy.deepcopy(edges)
        reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(nodes, nodes_snapshot, "the caller's nodes were mutated")
        self.assertEqual(edges, edges_snapshot, "the caller's edges were mutated")

    def test_the_survivor_provenance_list_is_not_shared_with_the_caller(self) -> None:
        """The surviving node must own its provenance list, not alias the input.

        A shared list is invisible today (every node in a call carries the same item
        id, so the dedupe never appends twice) but would leak a mutation into the
        caller's dicts as soon as two ids differ. Pin the copy by mutating the
        returned node and checking the input is unaffected.
        """
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        merged, _, _ = reconcile_actor_nodes(nodes, edges, row)
        merged[0]["provenance_item_ids"].append("SENTINEL")
        for node in nodes:
            self.assertNotIn("SENTINEL", node.get("provenance_item_ids") or [],
                             "the returned node shares its provenance list with the input")

    def test_a_target_with_no_node_is_materialised_not_dropped(self) -> None:
        """An entity the row resolved but no edge named must still exist as a node.

        Otherwise the merged edge would point at a nonexistent endpoint and the graph
        would stop being closed. This is the path that reaches the synthetic-node
        branch.
        """
        row = row_with(
            record("Petteri Orpo", "Petteri Orpo", "FI-ORPO"),
            record("Sanna Marin", "Sanna Marin", "FI-MARIN"),
        )
        nodes, edges = graph_from_edges(
            [edge_of("Orpo", "Something")], row
        )
        self.assertFalse(any(n["node_id"] == "entity:FI-ORPO" for n in nodes))
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        ids = {n["node_id"] for n in merged}
        self.assertIn("entity:FI-ORPO", ids)
        self.assertEqual(report["merged_count"], 1)
        for edge in merged_edges:
            self.assertIn(edge["source"], ids)
            self.assertIn(edge["target"], ids)
        materialised = [n for n in merged if n.get("materialised_from")]
        self.assertEqual(len(materialised), 1)

    def test_it_does_not_append_to_the_caller_node_list(self) -> None:
        """Materialising an absent target must not extend the caller's list."""
        row = row_with(
            record("Petteri Orpo", "Petteri Orpo", "FI-ORPO"),
            record("Sanna Marin", "Sanna Marin", "FI-MARIN"),
        )
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        length_before = len(nodes)
        reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(nodes), length_before)

    def test_provenance_item_ids_are_folded_on_merge(self) -> None:
        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        merged, _, _ = reconcile_actor_nodes(nodes, edges, row)
        # the provenance id is the canonical item id, which includes the country
        self.assertTrue(any(str(i).startswith("ITEM-1") for i in merged[0]["provenance_item_ids"]),
                        merged[0]["provenance_item_ids"])

    def test_an_empty_graph_reconciles_to_an_empty_graph(self) -> None:
        merged, merged_edges, report = reconcile_actor_nodes([], [], row_with())
        self.assertEqual(merged, [])
        self.assertEqual(merged_edges, [])
        self.assertEqual(report["merged_count"], 0)

    def test_a_malformed_resolution_cell_does_not_raise(self) -> None:
        row = {"video_id": "X", "ep24_entity_resolution_json": "{not json"}
        nodes, edges = graph_from_edges([edge_of("A", "B")], row)
        merged, _, report = reconcile_actor_nodes(nodes, edges, row)
        self.assertEqual(len(merged), 2)
        self.assertEqual(report["merged_count"], 0)


class SummaryVisibilityTests(unittest.TestCase):
    """Remaining fragmentation must appear in the summary text, not shape it silently."""

    def test_summary_reports_the_reconciliation_counters(self) -> None:
        from ep24_sna import graph_summary

        row = row_with(ORPO)
        nodes, edges = graph_from_edges([edge_of("Petteri Orpo", "Orpo", "FI-ORPO")], row)
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        metrics = apply_reconciliation_metrics(basic_metrics(merged, merged_edges), report)
        summary = graph_summary(merged, merged_edges, metrics)
        self.assertIn("reconciled_actors=1", summary)
        self.assertIn("ambiguous_actors=0", summary)
        self.assertIn("unresolved_actors=0", summary)

    def test_summary_reports_unresolved_actors_when_there_is_no_match(self) -> None:
        from ep24_sna import graph_summary

        row = row_with()
        nodes, edges = graph_from_edges([edge_of("A", "B")], row)
        merged, merged_edges, report = reconcile_actor_nodes(nodes, edges, row)
        metrics = apply_reconciliation_metrics(basic_metrics(merged, merged_edges), report)
        summary = graph_summary(merged, merged_edges, metrics)
        self.assertIn("unresolved_actors=2", summary)


if __name__ == "__main__":
    unittest.main()
