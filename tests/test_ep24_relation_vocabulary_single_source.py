"""Guard issue #92: EP24 electoral relation vocabulary has one source of truth."""

from __future__ import annotations

import ast
from pathlib import Path

from roihu_memory import RELATION_TYPES


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ("has_member", "candidate_on", "member_of_list", "eu_group")


def _relation_type_definitions() -> list[str]:
    definitions: list[str] = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts or ".git" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "RELATION_TYPES" for target in targets):
                definitions.append(path.relative_to(ROOT).as_posix())
    return sorted(definitions)


def test_electoral_relation_vocabulary_is_bare_predicates() -> None:
    assert RELATION_TYPES == EXPECTED
    assert all(" " not in predicate and "--" not in predicate for predicate in RELATION_TYPES)


def test_electoral_relation_vocabulary_is_defined_only_in_roihu_memory() -> None:
    assert _relation_type_definitions() == ["roihu_memory.py"]


def test_duplicate_relation_modules_do_not_return() -> None:
    assert not (ROOT / "ep24_lists.py").exists()
    assert not (ROOT / "ep24_relations.py").exists()
