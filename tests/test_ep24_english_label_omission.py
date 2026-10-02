"""Regression tests: an entry with no language metadata must not be exempt by omission.

Issue #101. Complements ``tests/test_ep24_english_label_coverage.py`` (PR #108),
which established the policy, the coverage metrics and the strict gate. This file
pins the case #108's language-gated rule could not see: entries that carry **no
``language`` at all**.

Why it matters, measured on the real books: the ``common`` codebook layer records
no ``language`` at file or entry level. Under a ``source_languages``-only rule
those entries returned ``required = False``, so they were not present, not
missing, and not counted as exemptions either — 2694 entries per country simply
uncounted. 26 of them are genuinely non-English (``Rassemblement National``,
``Sinn Féin``, ``Moderaterna``) and were therefore invisible to QA.

Synthetic fixtures only; no private codebook content.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402
from roihu_codebooks import (  # noqa: E402
    CodebookEntry,
    english_label_coverage,
    english_label_required,
)


def _entry(label, **kw):
    return CodebookEntry(entry_id=kw.pop("entry_id", label), kind=kw.pop("kind", "entity"),
                         label=label, **kw)


class TestNoLanguageIsNotAnExemption:
    def test_no_language_plus_non_english_label_is_required(self):
        """The hole: this returned False before the fix."""
        entry = _entry("Rassemblement National")
        assert entry.source_languages == []
        assert english_label_required(entry) is True

    def test_no_language_plus_english_label_is_not_required(self):
        """An already-English label has nothing to translate."""
        assert english_label_required(_entry("Abortion")) is False
        assert english_label_required(_entry("Accessibility and inclusivity")) is False

    def test_no_language_plus_person_name_is_not_required(self):
        """The English form is the same string, so a gloss is not required."""
        for label in ("Pedro Sánchez", "Björn Höcke", "François Hollande"):
            assert english_label_required(_entry(label)) is False, label

    def test_no_language_plus_handle_or_url_is_not_required(self):
        for label in ("@fundacjawosp", "http://example.org", "https://example.org/x"):
            assert english_label_required(_entry(label)) is False, label

    def test_documented_exemption_still_wins(self):
        entry = _entry("Rassemblement National",
                       metadata={"english_label_exempt_reason": "language-neutral acronym"})
        assert english_label_required(entry) is False


class TestOrganisationsAreNotExemptedByShape:
    """Two capitalised words is not enough to be a person name."""

    def test_organisations_are_required(self):
        for label in ("Les Républicains", "Fianna Fáil", "Partido Socialista",
                      "Rassemblement National", "Moderaterna", "Sinn Féin"):
            assert english_label_required(_entry(label)) is True, label


class TestLanguageBearingEntriesUnchanged:
    """The pre-existing rule must keep working exactly as before."""

    def test_non_english_language_is_required(self):
        assert english_label_required(_entry("Puolue", source_languages=["fi"])) is True

    def test_english_language_is_not_required(self):
        assert english_label_required(_entry("Party", source_languages=["en"])) is False

    def test_present_label_is_counted_present(self):
        qa = english_label_coverage([_entry("Partia Razem", source_languages=["pl"],
                                            english_label="Together Party")])
        assert qa["required_count"] == 1
        assert qa["present_count"] == 1
        assert qa["missing_count"] == 0
        assert qa["state"] == "PASS"


class TestCoverageCountsThePreviouslyInvisibleEntries:
    def test_common_layer_entries_enter_the_required_count(self):
        entries = [
            _entry("Rassemblement National"),          # no language, non-English -> required
            _entry("Abortion"),                        # no language, English -> exempt
            _entry("Pedro Sánchez"),                   # no language, person -> exempt
            _entry("Partia Razem", source_languages=["pl"]),  # language-bearing -> required
        ]
        qa = english_label_coverage(entries)
        assert qa["required_count"] == 2, "the language-less non-English entry must be counted"
        assert qa["missing_count"] == 2
        assert qa["state"] == "REVIEW_REQUIRED"

    def test_coverage_is_not_silently_100_percent(self):
        """The failure mode #101 describes: a gap that reads as complete."""
        qa = english_label_coverage([_entry("Rassemblement National")])
        assert qa["coverage_pct"] == 0.0
        assert qa["state"] == "REVIEW_REQUIRED"


class TestEndToEndThroughLoadProfile:
    def test_load_profile_reports_language_less_gaps(self, tmp_path):
        """Wires the policy through the real loader path."""
        codebooks = tmp_path / "codebooks"
        codebooks.mkdir()
        # a common book with no `language` key at all — the real shape
        (codebooks / "ep24_common_private.json").write_text(json.dumps({
            "schema": "ep24-private-codebook-v4",
            "country_code": "COMMON",
            "entries": [
                {"kind": "entity", "label": "Rassemblement National"},
                {"kind": "topic", "label": "Abortion"},
            ],
        }), encoding="utf-8")
        (codebooks / "ep24_poland_private.json").write_text(json.dumps({
            "schema": "ep24-private-codebook-v4",
            "country_code": "PL",
            "language": "pl",
            "entries": [{"kind": "entity", "label": "Partia Razem"}],
        }), encoding="utf-8")

        from roihu_codebooks import load_profile

        entries, meta = load_profile(tmp_path, "PL")
        qa = meta["english_label_coverage"]
        # required: the common non-English label + the country-layer Polish label
        assert qa["required_count"] == 2, qa
        assert qa["missing_count"] == 2
        assert qa["state"] == "REVIEW_REQUIRED"
        # the already-English common label is not counted as a gap
        assert "Abortion" not in qa["missing_entry_ids"]


def _all_country_fixture(tmp_path: Path) -> Path:
    """A codebook root that satisfies every entry in COUNTRY_PROFILES.

    ``bilingual_coverage_report`` iterates all ten countries, so a report fixture
    needs a book for each. Only the PL book carries the gap under test; the rest
    are minimal and already-English so they do not add to the totals.
    """
    from roihu_codebooks import COUNTRY_PROFILES

    codebooks = tmp_path / "codebooks"
    codebooks.mkdir(exist_ok=True)
    (codebooks / "ep24_common_private.json").write_text(json.dumps({
        "schema": "ep24-private-codebook-v4",
        "country_code": "COMMON",
        "entries": [{"kind": "entity", "label": "Rassemblement National"}],
    }), encoding="utf-8")
    for code, meta in COUNTRY_PROFILES.items():
        (codebooks / meta["file"]).write_text(json.dumps({
            "schema": "ep24-private-codebook-v4",
            "country_code": code,
            "language": meta["languages"][0],
            "entries": [{"kind": "entity", "label": "Partia Razem"}] if code == "PL"
                       else [{"kind": "topic", "label": "Abortion"}],
        }), encoding="utf-8")
    return tmp_path


class TestCoverageReportAndEnforcement:
    """Ported from the retired parallel implementation (issue #116).

    ``bilingual_coverage_report`` / ``assert_bilingual_coverage`` are kept but now
    derive from the canonical policy, so these pin that they still work and that
    the two entry points cannot drift again.
    """

    def test_report_and_coverage_agree(self, tmp_path):
        """The regression guard for #116: one policy, one answer."""
        report = cb.bilingual_coverage_report(_all_country_fixture(tmp_path))
        for country in report["countries"]:
            entries, meta = cb.load_profile(tmp_path, country["country_code"])
            assert country["missing_english"] == meta["english_label_coverage"]["missing_count"], (
                "the report and the canonical coverage metric must not disagree"
            )

    def test_report_has_the_required_shape(self, tmp_path):
        report = cb.bilingual_coverage_report(_all_country_fixture(tmp_path))
        assert report["kind"] == "ep24.bilingual_coverage/1"
        assert "policy" in report
        assert {"entries", "needs_english", "missing_english"} <= set(report["totals"])

    def test_report_breaks_down_by_layer(self, tmp_path):
        pl = next(c for c in cb.bilingual_coverage_report(_all_country_fixture(tmp_path))["countries"]
                  if c["country_code"] == "PL")
        assert pl["missing_by_layer"].get("common", 0) >= 1

    def test_a_large_gap_can_fail_loudly(self, tmp_path):
        report = cb.bilingual_coverage_report(_all_country_fixture(tmp_path))
        with pytest.raises(ValueError, match="bilingual coverage below threshold"):
            cb.assert_bilingual_coverage(report, max_missing_pct=0.0)

    def test_default_is_not_a_hard_block(self, tmp_path):
        """The pipeline keeps running while labels are being repaired."""
        cb.assert_bilingual_coverage(cb.bilingual_coverage_report(_all_country_fixture(tmp_path)))

    def test_failure_message_names_the_offending_country(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            cb.assert_bilingual_coverage(
                cb.bilingual_coverage_report(_all_country_fixture(tmp_path)), max_missing_pct=0.0)
        assert "PL=" in str(exc.value)


class TestSinglePolicyNoDrift:
    """The #116 regression guard.

    Two parallel predicates over the same corpus produced 4914 vs 889 for the same
    question, exposed as two CLI tools giving different answers. This pins that the
    module exposes exactly ONE policy predicate and one derived report.
    """

    def test_only_one_policy_predicate_exists(self):
        import roihu_codebooks as mod
        preds = [n for n in dir(mod)
                 if n.startswith("entry_") and "english" in n]
        assert preds == ["entry_needs_english_label"] or preds == [], preds
        # the canonical name must exist, the retired parallel name must not
        assert hasattr(mod, "english_label_required")
        assert not hasattr(mod, "entry_needs_english_label")

    def test_no_duplicate_definitions_in_the_module(self):
        import re
        from pathlib import Path
        src = Path(cb.__file__).read_text(encoding="utf-8")
        for name in ("label_looks_english", "_looks_like_person_name",
                     "_NON_ENGLISH_MARKERS", "english_label_required"):
            defs = len(re.findall(rf"^(?:def |){name}", src, re.M))
            assert defs <= 1, f"{name} is defined {defs} times"
