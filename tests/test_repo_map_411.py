"""The repository map must describe the tree it is committed alongside (#411).

A generated map is only worth having if it cannot silently go stale. `--check` answers that for a human;
this test answers it for CI, so a module added, removed, renamed or re-documented fails here instead of
turning the documentation into a lie nobody notices.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "repo_map.py"

sys.path.insert(0, str(ROOT / "tools"))
import repo_map  # noqa: E402


class TestTheCommittedMap:
    def test_the_committed_map_is_current(self):
        result = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=ROOT,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            "docs/REPO_MAP.md no longer matches the tree — run `python tools/repo_map.py --write`\n"
            + result.stdout + result.stderr
        )

    def test_the_map_names_every_pipeline_module(self):
        import json

        data = repo_map.build(ROOT)
        committed = (ROOT / "docs" / "REPO_MAP.md").read_text(encoding="utf-8")
        for module in data["modules"]:
            assert f"`{module['path']}`" in committed, module["path"]

    def test_the_stage_contract_is_the_source_of_truth_for_stage_order(self):
        # The guide may explain the stages; it may not invent a different order than the contract.
        from ep24_stage_contract import STAGE_CONTRACT

        guide = (ROOT / "docs" / "RESEARCHER_GUIDE.md").read_text(encoding="utf-8")
        for stage in STAGE_CONTRACT:
            assert stage.entry_point in guide, stage.entry_point


    def test_the_digest_ignores_the_duplicate_table_but_not_the_structure(self):
        """The check must survive a Python version that rounds a similarity score differently, and must
        still fail when a module is added. That is the whole reason a digest exists."""
        data = repo_map.build(ROOT)
        baseline = repo_map.structure_digest(data)
        noisy = json.loads(json.dumps(data))
        noisy["duplicates"] = [{"name": "x", "similarity": 0.123, "left": "a", "left_line": 1,
                                "right": "b", "right_line": 2, "lines": 10}]
        assert repo_map.structure_digest(noisy) == baseline
        changed = json.loads(json.dumps(data))
        changed["modules"].append({"path": "new_module.py", "lines": 1, "docstring": None,
                                   "role": "", "layer": "unclassified", "imported_by": 0,
                                   "is_entry_point": False, "functions_without_docstring": [],
                                   "prompt_constants": [], "unimported": True})
        assert repo_map.structure_digest(changed) != baseline

class TestTheScan:
    def _tree(self, tmp_path: Path, files: dict[str, str]) -> Path:
        for name, source in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return tmp_path

    def test_a_module_without_a_docstring_is_listed(self, tmp_path):
        repo_map.SKIP_DIRS.add("unused")  # keep the shared constant honest under test
        self._tree(tmp_path, {"stage.py": "def f():\n    return 1\n",
                              "documented.py": '"""What this does."""\n'})
        data = repo_map.build(tmp_path)
        by_path = {m["path"]: m for m in data["modules"]}
        assert by_path["stage.py"]["docstring"] is None
        assert by_path["documented.py"]["docstring"] == "What this does."

    def test_tests_are_counted_separately_not_mapped(self, tmp_path):
        self._tree(tmp_path, {"stage.py": '"""x"""\n', "tests/test_a.py": '"""t"""\n'})
        data = repo_map.build(tmp_path)
        assert all(not m["path"].startswith("tests/") for m in data["modules"])
        assert data["tests"] == ["test_a.py"]

    def test_a_script_is_its_own_layer(self, tmp_path):
        self._tree(tmp_path, {"scripts/audit_thing.py": '"""audit"""\n'})
        data = repo_map.build(tmp_path)
        assert data["modules"][0]["layer"] == "operator script"

    def test_the_stage_wrapper_prefix_is_an_entry_point_layer(self, tmp_path):
        self._tree(tmp_path, {"step_4_roihu_summary.py": '"""Step 4."""\n'
                                                          'if __name__ == "__main__":\n    raise SystemExit(0)\n'})
        module = repo_map.build(tmp_path)["modules"][0]
        assert module["layer"] == "entry point"
        assert module["is_entry_point"] is True

    def test_identical_function_bodies_in_two_modules_are_found(self, tmp_path):
        # At least ten lines: shorter bodies are not compared, because two three-line helpers that look
        # alike are not evidence of duplication.
        body = ("def build_context(record, codebook):\n"
                "    parts = []\n"
                "    for key in sorted(codebook):\n"
                "        value = record.get(key, '')\n"
                "        if value:\n"
                "            parts.append(value)\n"
                "    joined = ' '.join(parts)\n"
                "    cleaned = joined.strip()\n"
                "    normalised = ' '.join(cleaned.split())\n"
                "    return normalised or None\n")
        self._tree(tmp_path, {"step_six.py": '"""s6"""\n' + body,
                              "step_seven.py": '"""s7"""\n' + body})
        pairs = repo_map.build(tmp_path)["duplicates"]
        assert len(pairs) == 1
        assert pairs[0]["similarity"] == 1.0
        assert {pairs[0]["left"], pairs[0]["right"]} == {"step_six.py", "step_seven.py"}

    def test_functions_that_differ_are_not_reported(self, tmp_path):
        self._tree(tmp_path, {
            "a.py": '"""a"""\ndef f(x):\n    return x + 1\n',
            "b.py": '"""b"""\ndef f(x):\n    return [item for item in x if item]\n',
        })
        assert repo_map.build(tmp_path)["duplicates"] == []

    def test_short_functions_are_not_compared(self, tmp_path):
        self._tree(tmp_path, {"a.py": '"""a"""\ndef f():\n    return 1\n',
                              "b.py": '"""b"""\ndef f():\n    return 1\n'})
        assert repo_map.build(tmp_path)["duplicates"] == []

    def test_a_prompt_constant_is_detected_and_a_short_string_is_not(self, tmp_path):
        long_prompt = "You are an analyst. " + ("Return JSON with fields. " * 40)
        self._tree(tmp_path, {"stage.py": f'"""s"""\nSYSTEM_PROMPT = {long_prompt!r}\nBANNER = "hi"\n'})
        module = repo_map.build(tmp_path)["modules"][0]
        assert module["prompt_constants"] == ["SYSTEM_PROMPT"]

    def test_an_unimported_module_is_flagged_and_an_imported_one_is_not(self, tmp_path):
        self._tree(tmp_path, {
            "used.py": '"""u"""\n',
            "orphan.py": '"""o"""\n',
            "importer.py": '"""i"""\nimport used\n',
        })
        by_path = {m["path"]: m for m in repo_map.build(tmp_path)["modules"]}
        assert by_path["orphan.py"]["unimported"] is True
        assert by_path["used.py"]["unimported"] is False

    def test_the_rendered_map_is_deterministic(self, tmp_path):
        self._tree(tmp_path, {"a.py": '"""a"""\n', "b.py": '"""b"""\n'})
        first = repo_map.render(repo_map.build(tmp_path))
        second = repo_map.render(repo_map.build(tmp_path))
        assert first == second
