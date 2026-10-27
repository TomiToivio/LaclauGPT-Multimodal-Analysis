"""Tests for the public-repository privacy gate (#306, deliverable 4).

The gate is only worth having if it fails on a real leak and stays quiet on documentation. Both
directions are tested: a gate that never fires is theatre, and a gate that always fires is ignored.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "ep24" / "privacy_gate.py"


#: Assembled at runtime: a literal password in the source gets rewritten by secret scanners (and by
#: some editors), which would silently turn this fixture into a placeholder and make the test pass
#: for the wrong reason.
_SECRET_URI = "mongodb://bob:" + "q7" * 3 + "@db.example.com/x"


def _load():
    spec = importlib.util.spec_from_file_location("privacy_gate", SPEC)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["privacy_gate"] = module
    spec.loader.exec_module(module)
    return module


gate = _load()


class TestPlaceholders:
    @pytest.mark.parametrize("uri", [
        "mongodb://...",
        "mongodb://user:***@host/db",
        "mongodb://example.invalid/research_db",
        "mongodb://synthetic.invalid:27017",
        "mongodb://secret-host.invalid:27017/?tls=true",
        "mongodb://localhost:27017/ep24",
        "mongodb://x",
        "mongodb://h/db",
    ])
    def test_documented_placeholders_are_not_credentials(self, uri):
        assert gate.looks_like_placeholder(uri) is True

    @pytest.mark.parametrize("uri", [
        "mongodb://realuser:hunter2@db.example.com:27017/ep24",
        "mongodb+srv://analyst:s3cret@cluster0.abcd.mongodb.net/prod",
        "mongodb://10.4.5.6:27017/ep24",
    ])
    def test_a_real_looking_credential_is_still_reported(self, uri):
        assert gate.looks_like_placeholder(uri) is False


class TestScanFile:
    def _scan(self, tmp_path: Path, text: str, name: str = "thing.py"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return gate.scan_file(path, tmp_path)

    def test_a_real_credential_is_critical(self, tmp_path):
        findings = self._scan(tmp_path, 'URI = "mongodb://bob:hunter2@db.example.com/x"\n')
        assert [f.rule for f in findings] == ["mongodb-uri"]
        assert findings[0].severity == "critical"

    def test_a_documented_placeholder_is_silent(self, tmp_path):
        assert self._scan(tmp_path, 'URI = "mongodb://user:***@host/db"\n') == []

    def test_a_guard_that_forbids_a_private_path_is_not_a_leak(self, tmp_path):
        # The rule that enforces the rule has to contain the pattern.
        text = 'FORBIDDEN = re.compile(r"/mnt/workspace/")\nassert "/mnt/workspace" not in text\n'
        assert self._scan(tmp_path, text) == []

    def test_a_plain_private_path_is_still_reported(self, tmp_path):
        findings = self._scan(tmp_path, 'ROOT = "/mnt/workspace/NooPunk"\n')
        assert [f.rule for f in findings] == ["private-path"]
        assert findings[0].severity == "high"

    def test_a_git_ssh_remote_is_not_an_email(self, tmp_path):
        assert self._scan(tmp_path, 'ORIGIN = "git@github.com:owner/repo.git"\n') == []

    def test_a_published_contact_address_is_reported_but_not_critical(self, tmp_path):
        findings = self._scan(tmp_path, "contact: someone@example.org\n")
        assert [f.severity for f in findings] == ["medium"]

    def test_private_hostnames_are_reported(self, tmp_path):
        findings = self._scan(tmp_path, "host laskin01\n")
        assert [f.rule for f in findings] == ["private-host"]


class TestExitPolicy:
    def test_medium_findings_do_not_block_by_default(self, tmp_path):
        (tmp_path / "a.py").write_text("contact: someone@example.org\n", encoding="utf-8")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(SPEC), "--root", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout

    def test_critical_findings_block(self, tmp_path):
        (tmp_path / "a.py").write_text('U = "mongodb://bob:hunter2@db.example.com/x"\n', encoding="utf-8")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(SPEC), "--root", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1, result.stdout
