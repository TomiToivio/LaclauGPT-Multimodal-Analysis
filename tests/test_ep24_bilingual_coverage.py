"""Bilingual ``english_label`` coverage policy and enforcement (issue #101).

Synthetic fixtures only — no private codebook content. These tests pin:

1. the policy rule: an entry needs an English label when its canonical label is
   not already English;
2. the regression that motivated the issue: the ``common`` layer carries no
   ``language`` metadata, so a language-gated test is blind to it and the
   reported gap is far smaller than the real one;
3. that coverage gaps can be turned into a loud failure (never silent);
4. that the enforcement is a signal, not a hard block, by default.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _entry(label, **kw):
    base = {"kind": "entity", "label": label}
    base.update(kw)
    return base


def _country_fixture(tmp_path: Path) -> Path:
    """A two-layer fixture with a `common` book that has no language metadata."""
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    # common layer: no file-level "language" — reproduces the real shape
    _write(
        codebooks / "ep24_common_private.json",
        {
            "schema": "ep24-private-codebook-v4",
            "country_code": "COMMON",
            "entries": [
                _entry("Abortion"),                       # already English -> exempt
                _entry("Democracia"),                     # topic, non-English -> needs
                _entry("Rassemblement National"),          # org, French -> needs
                _entry("Pedro Sánchez"),                   # person -> exempt
                _entry("@fundacjawosp"),                   # handle -> exempt
            ],
        },
    )
    # country layer: has a language, like the real country books
    _write(
        codebooks / "ep24_poland_private.json",
        {
            "schema": "ep24-private-codebook-v4",
            "country_code": "PL",
            "language": "pl",
            "entries": [
                _entry("Partia Razem"),                   # needs
                _entry("Abortion Rights"),                # already English -> exempt
                _entry("Adam Bielan"),                    # person -> exempt
            ],
        },
    )
    return tmp_path


class TestPolicy:
    def test_already_english_labels_are_exempt(self):
        assert cb.label_looks_english("Abortion Rights")
        assert cb.label_looks_english("Accessibility and inclusivity")
        assert not cb.entry_needs_english_label(
            cb.CodebookEntry(entry_id="a", kind="topic", label="Abortion")
        )

    def test_non_english_labels_need_a_gloss(self):
        for label in ("Partia Razem", "Rassemblement National", "Democracia",
                      "Bündnis 90/Die Grünen", "Vasemmistoliitto"):
            assert not cb.label_looks_english(label), label
            assert cb.entry_needs_english_label(
                cb.CodebookEntry(entry_id="x", kind="entity", label=label)
            ), label

    def test_handles_and_urls_are_exempt(self):
        for label in ("@fundacjawosp", "http://example.org", "https://example.org/x"):
            assert not cb.entry_needs_english_label(
                cb.CodebookEntry(entry_id="x", kind="entity", label=label)
            ), label

    def test_plain_person_names_are_exempt(self):
        """An English form would be the same string, so flagging them is noise."""
        for label in ("Pedro Sánchez", "Björn Höcke", "Karel Novák"):
            assert not cb.entry_needs_english_label(
                cb.CodebookEntry(entry_id="x", kind="entity", label=label)
            ), label

    def test_organisations_are_not_mistaken_for_person_names(self):
        """Shape alone is not enough: these are two capitalised words but bodies."""
        for label in ("Les Républicains", "Fianna Fáil", "Partido Socialista",
                      "Rassemblement National", "Sinn Féin"):
            assert cb.entry_needs_english_label(
                cb.CodebookEntry(entry_id="x", kind="entity", label=label)
            ), label


class TestCommonLayerIsNotInvisible:
    """The regression from #101: a language gate cannot see the common layer."""

    def test_common_layer_has_no_source_languages(self, tmp_path):
        root = _country_fixture(tmp_path)
        entries, _meta = cb.load_profile(root, "PL")
        common = [e for e in entries if e.layer == "common"]
        assert common, "fixture must contain a common layer"
        assert all(not e.source_languages for e in common), (
            "common-layer entries carry no language metadata — this is why the "
            "old gate was blind to them"
        )

    def test_policy_sees_common_layer_entries_a_language_gate_would_miss(self, tmp_path):
        """The language gate is blind to the common layer, so it misses real gaps.

        On this fixture the old gate and the new policy happen to return the same
        *count* (3) while flagging entirely different entries — which is the
        point: equality of the number hides that the gate caught already-English
        labels and missed the real common-layer gaps.
        """
        root = _country_fixture(tmp_path)
        entries, _meta = cb.load_profile(root, "PL")

        old_gate = {
            e.label for e in entries
            if e.source_languages and any(lang != "en" for lang in e.source_languages)
            and not e.english_label
        }
        new_policy = {
            e.label for e in entries
            if cb.entry_needs_english_label(e) and not e.english_label
        }

        # the policy catches common-layer gaps the gate cannot see
        assert "Democracia" in new_policy
        assert "Rassemblement National" in new_policy
        assert "Democracia" not in old_gate

        # and it no longer flags labels that are already English
        assert "Abortion Rights" not in new_policy
        assert "Abortion Rights" in old_gate, (
            "the fixture must show the gate's false positive for this test to mean anything"
        )
        assert "Adam Bielan" not in new_policy  # person name, exempt

    def test_common_layer_entries_are_attributed_not_dropped(self, tmp_path):
        root = _country_fixture(tmp_path)
        pl = next(c for c in cb.bilingual_coverage_report(root)["countries"]
                  if c["country_code"] == "PL")
        assert pl["missing_by_layer"].get("common", 0) >= 2

    def test_metric_itself_uses_the_policy_not_a_language_gate(self, tmp_path):
        """Pins `load_profile`'s metric directly, so reverting it fails here.

        The gate and the policy can agree on a *count* while disagreeing on
        *which* entries — so this asserts on identity, not length. Without this
        test, reverting the metric to `source_languages`-gating silently passes
        the rest of the suite.
        """
        root = _country_fixture(tmp_path)
        entries, meta = cb.load_profile(root, "PL")

        reported = set(meta["missing_english_entry_ids"])
        common_gap = {e.entry_id for e in entries
                      if e.layer == "common"
                      and cb.entry_needs_english_label(e)
                      and not e.english_label}
        already_english = {e.entry_id for e in entries
                           if e.label == "Abortion Rights"}

        assert common_gap, "fixture must contain a common-layer gap"
        assert common_gap <= reported, (
            "the metric must report common-layer gaps, which a language gate cannot see"
        )
        assert not (already_english & reported), (
            "the metric must not report already-English labels as missing"
        )


class TestCoverageReport:
    def test_report_has_the_required_shape(self, tmp_path):
        root = _country_fixture(tmp_path)
        report = cb.bilingual_coverage_report(root)
        assert report["kind"] == "ep24.bilingual_coverage/1"
        assert "policy" in report
        assert {"entries", "needs_english", "missing_english", "missing_pct"} <= set(
            report["totals"]
        )

    def test_report_counts_only_entries_that_need_a_gloss(self, tmp_path):
        root = _country_fixture(tmp_path)
        report = cb.bilingual_coverage_report(root)
        pl = next(c for c in report["countries"] if c["country_code"] == "PL")
        # flagged: Democracia, Rassemblement National (common), Partia Razem (country)
        assert pl["needs_english"] == 3
        assert pl["missing_english"] == 3
        # everything else in the fixture is exempt, so the denominator is larger
        assert pl["entries"] == 8

    def test_report_breaks_down_by_layer(self, tmp_path):
        root = _country_fixture(tmp_path)
        pl = next(c for c in cb.bilingual_coverage_report(root)["countries"]
                  if c["country_code"] == "PL")
        assert pl["missing_by_layer"]["common"] == 2
        assert pl["missing_by_layer"]["country"] == 1

    def test_report_lists_entry_ids_for_action(self, tmp_path):
        root = _country_fixture(tmp_path)
        pl = next(c for c in cb.bilingual_coverage_report(root)["countries"]
                  if c["country_code"] == "PL")
        assert len(pl["missing_entry_ids"]) == pl["missing_english"]


class TestEnforcement:
    def test_a_large_gap_can_fail_loudly(self, tmp_path):
        report = cb.bilingual_coverage_report(_country_fixture(tmp_path))
        with pytest.raises(ValueError, match="bilingual coverage below threshold"):
            cb.assert_bilingual_coverage(report, max_missing_pct=1.0)

    def test_a_generous_threshold_passes(self, tmp_path):
        report = cb.bilingual_coverage_report(_country_fixture(tmp_path))
        cb.assert_bilingual_coverage(report, max_missing_pct=100.0)  # must not raise

    def test_default_is_not_a_hard_block(self, tmp_path):
        """The pipeline keeps running while labels are being repaired."""
        report = cb.bilingual_coverage_report(_country_fixture(tmp_path))
        cb.assert_bilingual_coverage(report)  # must not raise

    def test_failure_message_names_the_offending_countries(self, tmp_path):
        report = cb.bilingual_coverage_report(_country_fixture(tmp_path))
        with pytest.raises(ValueError) as exc:
            cb.assert_bilingual_coverage(report, max_missing_pct=0.0)
        assert "PL=" in str(exc.value)
