"""The repository map must name every top-level module, and every stage must be in the map (#411).

A repository map is useful exactly until the first module is added and nobody updates it. This test makes
that visible in CI: the map and the tree cannot disagree. It is deliberately structural — it asserts that
documentation covers the code, not what the documentation says.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs" / "REPOSITORY_MAP.md"

#: Modules that exist but intentionally need no row of their own, with the reason.
EXEMPT = {
    "conftest.py": "pytest fixtures; listed by its purpose, not by name",
}

#: Directories whose Python files are covered by a directory-level row instead of one row per file.
DIRECTORY_COVERED = ("tests", "scripts", "examples", "experiments")


def top_level_modules() -> set[str]:
    return {
        path.name
        for path in ROOT.glob("*.py")
        if path.is_file() and not path.name.startswith("_")
    }


def map_text() -> str:
    assert MAP.exists(), f"{MAP.name} is missing; the repository has no map"
    return MAP.read_text(encoding="utf-8")


def test_every_top_level_module_is_named_in_the_map():
    named = map_text()
    missing = sorted(
        module for module in top_level_modules()
        if module not in named and module not in EXEMPT
    )
    assert not missing, (
        "These top-level modules are not named in docs/REPOSITORY_MAP.md, so a reader cannot find them:\n  "
        + "\n  ".join(missing)
        + "\nAdd a row describing what each one is for, or extend the EXEMPT list with a reason."
    )


def test_every_stage_entry_point_appears_in_the_map():
    from ep24_stage_contract import STAGE_CONTRACT

    named = map_text()
    missing = [
        stage.entry_point for stage in STAGE_CONTRACT if stage.entry_point not in named
    ]
    assert not missing, (
        "Stage entry points missing from the repository map: " + ", ".join(missing)
    )


def test_the_stage_table_matches_the_contract():
    """The map's step numbers and entry points must agree with `ep24_stage_contract.py`."""
    from ep24_stage_contract import STAGE_CONTRACT

    named = map_text()
    for stage in STAGE_CONTRACT:
        pattern = re.compile(
            rf"^\|\s*{stage.number}\s*\|\s*`{re.escape(stage.name)}`\s*\|\s*`{re.escape(stage.entry_point)}`",
            re.MULTILINE,
        )
        assert pattern.search(named), (
            f"The map's stage table has no row for step {stage.number} ({stage.name} -> "
            f"{stage.entry_point}); regenerate it from the contract instead of editing it by hand."
        )


def test_a_documented_module_does_not_silently_vanish():
    """A module named in the map that no longer exists is a broken map, not a broken codebase."""
    named = map_text()
    referenced = set(re.findall(r"`([A-Za-z0-9_]+\.py)`", named))
    # Files outside the repository root (or in subdirectories) are covered by their own rows.
    existing = {path.name for path in ROOT.rglob("*.py")}
    missing = sorted(
        name for name in referenced
        if name not in existing and not name.startswith("step_")
    )
    assert not missing, (
        "docs/REPOSITORY_MAP.md names files that do not exist: " + ", ".join(missing)
    )
