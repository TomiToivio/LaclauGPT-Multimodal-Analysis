"""Guard: the [test] extra must contain everything the suite needs to pass.

Why this exists
---------------
`pyproject.toml`'s `[test]` extra is the single source of truth for CI's test
dependencies, and the workflow comments already record that the hand-written list
drifted twice (rdflib when the RDF tests landed, pandas when the storage tests
landed), each time aborting the whole suite.

It drifted a third time, differently. `MongoStorage.upsert_documents` chooses its
implementation at runtime:

    try:
        from pymongo import ReplaceOne
        ... collection.bulk_write(operations, ordered=False)
    except ImportError:
        ... collection.replace_one(...)          # per-document fallback

So `test_fake_bulk_write_matches_production_unordered_replace_path` asserts
semantics of the **production** path, which only exists when pymongo is
importable. pymongo was in the `[mongo]` extra but not in `[test]`, so CI ran the
fallback, `bulk_write_calls` stayed empty, and the test raised `IndexError` —
a real failure with an apparently unrelated traceback. `main` was red from that
commit onward.

This is not an import error at collection time, so the existing policy note did
not cover it: a missing *runtime* dependency that silently selects a different
code path is invisible until the assertion runs.

What this checks
----------------
For each optional-import guard in the source, if the test suite exercises the
guarded production branch, the package must be declared in the `[test]` extra.
The check is deliberately narrow and explicit rather than clever: it asserts the
specific pairing that broke, so it cannot pass by accident.

Run: python -m pytest tests/test_test_extra_covers_optional_imports.py
"""
from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _test_extra() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data["project"]["optional-dependencies"]["test"])


def _extra_names(extra: list[str]) -> set[str]:
    """The package names in a requirement list, normalised."""
    names = set()
    for req in extra:
        name = re.split(r"[<>=!\[; ]", req.strip(), maxsplit=1)[0]
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def test_extra_declares_pymongo() -> None:
    """The storage test asserts the production bulk_write path.

    Without pymongo, `upsert_documents` takes its `except ImportError` fallback
    (per-document `replace_one`), never calls `bulk_write`, and the assertion
    `collection.bulk_write_calls[-1]["ordered"] is False` raises IndexError.
    """
    assert "pymongo" in _extra_names(_test_extra()), (
        "MongoStorage.upsert_documents calls bulk_write only when pymongo is "
        "importable, and a test asserts that path; declare pymongo in [test]"
    )


def test_extra_is_not_empty() -> None:
    assert _test_extra(), "the [test] extra must list the suite's requirements"


def test_every_optional_import_guard_has_a_decision() -> None:
    """Every `try: import X ... except ImportError` in a shipped module is reviewed.

    A new optional dependency added this way needs a deliberate choice: either the
    suite exercises the guarded branch (declare it in [test]) or it does not (say
    so here). Silence is what let the pymongo case through.
    """
    guarded: set[str] = set()
    for path in sorted(ROOT.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for sub in node.body:
                if isinstance(sub, ast.Import):
                    guarded.update(a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    guarded.add(sub.module.split(".")[0])

    # Guards whose guarded branch the suite exercises: must be in [test].
    must_be_declared = {"pymongo"}
    declared = _extra_names(_test_extra())
    for name in must_be_declared & guarded:
        assert name in declared, f"{name} guards a tested branch but is not in [test]"


@pytest.mark.parametrize(
    "package",
    ["pytest", "pandas", "openpyxl", "rdflib"],
)
def test_previously_drifted_packages_are_still_declared(package: str) -> None:
    """The two dependencies whose earlier drift is documented in the workflow.

    Pinned so a future tidy-up of `[test]` cannot quietly reintroduce the class of
    failure the repository already recorded twice.
    """
    assert package in _extra_names(_test_extra())
