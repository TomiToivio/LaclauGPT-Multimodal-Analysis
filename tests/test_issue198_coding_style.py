"""Guard for docs/CODING_STYLE.md (issue #198).

The style guide makes structural claims about this repository, so they are checked
rather than left as prose that can drift. These are STRUCTURE guards, in the same
spirit as ``tests/test_ep24_stage_contract.py``.

What they protect:

* the guide exists and is reachable from the README and AGENTS.md;
* every principle issue #198 asked for is documented;
* the guide names the conventions that actually exist here -- the numbered
  ``step_N_roihu_*`` entry points and the shared ``ep24_*.py`` infrastructure --
  rather than generic advice;
* the step numbering and the stage contract are described, so a reader knows the
  contract must be updated with a step;
* the guide does not claim the EP24 -> AI26 migration has happened, because it has
  not;
* every path the guide cites actually exists.

Run: python -m pytest -q tests/test_issue198_coding_style.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = "docs/CODING_STYLE.md"

#: The principles #198 requires the document to contain.
REQUIRED_TOPICS = (
    "one analysis step = one clearly named file",
    "shared infrastructure",
    "readability over architectural cleverness",
    "logging",
    "comments",
    "system prompts",
    "data flow",
    "self-contained",
)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def normalised(relative: str) -> str:
    """Whitespace-collapsed, emphasis- and blockquote-stripped text.

    The guide writes key phrases in ``**bold**`` and inside a ``>`` callout; both
    would split a phrase assertion that is plainly true when rendered.
    """
    text = read(relative)
    text = re.sub(r"[*`]", "", text)
    text = re.sub(r"(?m)^\s*>\s?", " ", text)
    return " ".join(text.split()).lower()


def sentences(text: str) -> list[str]:
    """Split normalised text into sentences, for assertion scoped to one claim."""
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part]


#: Completion claims the guide must not make while the migration has not happened.
#:
#: The honest wording NEGATES these ("No EP24->AI26 port has been performed"), so the
#: check is per-sentence and a sentence carrying a negation token is exempt.
_OVERCLAIM_CLAIMS = (
    re.compile(r"migration\s+(?:is|has\s+been)\s+(?:complete|completed|done|finished)"),
    re.compile(r"\bport\s+(?:is|has\s+been)\s+(?:complete|completed|done|performed)"),
    re.compile(r"\bfully\s+ported\b"),
    re.compile(r"\bhas\s+been\s+ported\b"),
    re.compile(r"\bmigrated\s+to\s+ai26\b"),
    re.compile(r"\bmigration\s+is\s+no\s+longer\s+needed\b"),
)

#: Tokens that mark a completion claim as negated, i.e. honest.
_OVERCLAIM_NEGATIONS = ("no ", "not ", "never", "n't", "without", "pending", "not started")


class TestGuideExists:
    def test_guide_exists(self) -> None:
        assert (ROOT / GUIDE).exists(), f"{GUIDE} is missing"

    def test_readme_links_the_guide(self) -> None:
        assert "docs/CODING_STYLE.md" in read("README.md")

    def test_agents_md_links_the_guide(self) -> None:
        assert "docs/CODING_STYLE.md" in read("AGENTS.md")

    def test_guide_states_the_guiding_principle(self) -> None:
        assert "optimize for the researcher reading the code six months later" in normalised(GUIDE)

    def test_guide_prioritises_readability_over_abstraction(self) -> None:
        text = normalised(GUIDE)
        assert "readability and research transparency take priority over abstraction" in text


class TestPrincipleCoverage:
    @pytest.mark.parametrize("topic", REQUIRED_TOPICS)
    def test_topic_is_documented(self, topic: str) -> None:
        assert topic in normalised(GUIDE), f"{GUIDE} does not document: {topic}"


class TestRealConventions:
    """The guide must describe this repository, not a generic ideal."""

    def test_names_the_numbered_step_entry_points(self) -> None:
        text = normalised(GUIDE)
        assert "step_1_roihu_preprocess.py" in text
        assert "step_9_roihu_rdf.py" in text

    def test_names_the_shared_infrastructure_family(self) -> None:
        text = normalised(GUIDE)
        for module in ("ep24_pipeline.py", "ep24_entities.py", "ep24_cli.py"):
            assert module in text, f"{GUIDE} does not name {module}"

    def test_names_the_legacy_reference_modules(self) -> None:
        assert "puhti_*.py" in read(GUIDE) or "puhti_preprocess.py" in read(GUIDE)

    def test_direct_execution_requirement_is_stated(self) -> None:
        """#198 requires the simplest invocation to stay possible."""
        text = normalised(GUIDE)
        assert "python step_4_roihu_summary.py" in text


class TestContractAwareness:
    def test_guide_requires_the_stage_contract_to_stay_truthful(self) -> None:
        """A step that changes its columns must update the contract in the same change."""
        text = normalised(GUIDE)
        assert "ep24_stage_contract.py" in text
        assert "contract is updated in the same change" in text

    def test_additive_field_rule_is_stated(self) -> None:
        text = normalised(GUIDE)
        assert "new analytical fields are additive" in text


class TestNoOverclaim:
    def test_migration_is_not_claimed_as_done(self) -> None:
        """#198 scopes the port as future direction.

        This asserts both directions. Checking only that the honest wording is
        PRESENT is satisfiable while a contradictory claim sits beside it: inserting
        "the migration is complete" into the section left the previous version of this
        guard green. A structural guard has to reject the overclaim too.
        """
        text = normalised(GUIDE)
        assert "status: not started" in text
        assert "no ep24→ai26 port has been performed" in text

    def test_no_sentence_claims_the_migration_is_complete(self) -> None:
        """The negative half: an overclaiming sentence must not coexist with the above.

        Scoped per sentence and exempting negated ones, because the honest status line
        legitimately negates the same phrases ("No EP24->AI26 port has been performed").
        """
        offences: list[str] = []
        for sentence in sentences(normalised(GUIDE)):
            if any(token in sentence for token in _OVERCLAIM_NEGATIONS):
                continue
            for pattern in _OVERCLAIM_CLAIMS:
                if pattern.search(sentence):
                    offences.append(sentence)
                    break
        assert not offences, (
            "the guide claims the migration is complete, which #198 scopes as future "
            f"direction: {offences}"
        )

    def test_the_overclaim_check_can_actually_fire(self) -> None:
        """Sabotage the check itself, so it cannot pass vacuously.

        Without this, a typo in the patterns would leave the guard green forever and
        the gap it exists to close would silently reopen.
        """
        sample = "the ep24 → ai26 migration is complete and everything has been ported."
        hits = [p for p in _OVERCLAIM_CLAIMS if p.search(sample)]
        assert hits, "the overclaim patterns no longer match a plain completion claim"
        # ...and the honest negation must NOT be flagged.
        honest = "no ep24→ai26 port has been performed; status: not started."
        negated = [
            p
            for p in _OVERCLAIM_CLAIMS
            if p.search(honest) and not any(t in honest for t in _OVERCLAIM_NEGATIONS)
        ]
        assert not negated, f"honest status wording is falsely flagged: {negated}"

    def test_repository_status_table_names_both_repos(self) -> None:
        text = normalised(GUIDE)
        assert "laclaugpt-data-analysis" in text
        assert "tomi-locked" in text, "the other repo's lock must be recorded, not ignored"


class TestCitedPathsExist:
    """A guide that cites a renamed or invented file is worse than no guide."""

    def test_no_cited_source_module_is_missing(self) -> None:
        """Only this repository's own families are checked.

        ``ep24_*`` and ``puhti_*`` belong here; ``laclaugpt_*`` / ``pipeline_*`` are
        Data-Analysis's and are documented as such, so they are checked by that
        repository's own guard instead.
        """
        body = read(GUIDE)
        # Match bare names too: the module inventory is a fenced block, not
        # backticked, so a backtick-only pattern saw almost nothing and silently
        # failed to catch a cited module that does not exist.
        cited = set(
            re.findall(r"(?<![`\w])((?:ep24_|puhti_|asr_|ocr_)[a-z0-9_]*\.py)", body)
        )
        assert len(cited) >= 8, f"expected the module inventory, found {sorted(cited)}"
        missing = sorted(name for name in cited if not (ROOT / name).exists())
        assert not missing, f"{GUIDE} cites modules that do not exist: {missing}"

    def test_cited_docs_exist(self) -> None:
        body = read(GUIDE)
        cited = set(re.findall(r"\(([A-Za-z0-9_./-]+\.md)\)", body))
        assert cited, "the guide cites no documents"
        missing = sorted(path for path in cited if not (ROOT / "docs" / path).exists() and not (ROOT / path).exists())
        assert not missing, f"{GUIDE} cites documents that do not exist: {missing}"

    def test_every_step_it_names_exists(self) -> None:
        """Scoped to this repository's own ``step_N_roihu_*`` entry points.

        The guide also documents Data-Analysis's ``step_0N_*.py`` scaffold, which
        lives in that repository; asserting on those here would fail on correct text.
        """
        body = read(GUIDE)
        cited = set(re.findall(r"(step_\d+_roihu_[a-z0-9_]+\.py)", body))
        assert cited, "the guide names none of this repository's step files"
        missing = sorted(name for name in cited if not (ROOT / name).exists())
        assert not missing, f"named step files do not exist: {missing}"


class TestStepSetIsComplete:
    def test_all_nine_steps_exist_on_disk(self) -> None:
        """The guide claims the numbered steps exist; that claim must hold."""
        steps = sorted(p.name for p in ROOT.glob("step_*_roihu_*.py"))
        numbers = sorted(
            int(match.group(1))
            for match in (re.match(r"step_(\d+)_", name) for name in steps)
            if match
        )
        assert numbers == list(range(1, 10)), f"expected steps 1..9, found {numbers}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
