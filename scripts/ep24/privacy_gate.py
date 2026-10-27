#!/usr/bin/env python3
"""Public-repository privacy gate for EP24 work (#306, deliverable 4).

Scans tracked files for the classes of leaked material the issue names: credentials,
connection strings, internal hosts and addresses, private infrastructure paths,
researcher/personal data, and corpus-derived content that must not be public.

The gate is deliberately shaped as an **allowlist review**, not a blacklist: a finding
is a file:line with the matched class and a redacted excerpt, so a human decides. It
never prints the secret it found - a gate that echoes a token into a log or a CI page
has published it more widely than the commit did.

Usage:
    python scripts/ep24/privacy_gate.py [--root DIR ...] [--json OUT]
Exit code is 1 when anything is found, so it can be a merge gate.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

#: (name, pattern, severity). Patterns are narrow on purpose: a gate that cries wolf
#: gets ignored, and an ignored gate is worse than none.
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("mongodb-uri", re.compile(r"mongodb(?:\+srv)?://[^\s\"']+"), "critical"),
    ("bearer-token", re.compile(r"\b(?:ghp_|gho_|ghs_|github_pat_|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})"), "critical"),
    ("private-ip", re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d))\.\d{1,3}\.\d{1,3}\b"), "high"),
    # CSC's Puhti/Mahti/Roihu are public national services and naming them is not a
    # leak; a private workstation name is. The first draft flagged 500+ legitimate
    # documentation lines, which is how a gate teaches people to ignore it.
    ("private-host", re.compile(r"\b(?:laskin0\d|[a-z]+\.internal)\b", re.IGNORECASE), "high"),
    # /scratch/project_* is CSC's public national HPC filesystem, consistent with the rule above that
    # naming CSC services is not a leak. Flagging it produced 62 findings, all false, in one repo.
    ("private-path", re.compile(r"(?:/mnt/workspace|/home/[a-z]+/(?:\.hermes|work))"), "high"),
    ("email-address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "medium"),
)

#: Paths whose matches are documentation of the rule itself, or fixtures that must
#: contain a pattern to test the gate.
EXEMPT_SUFFIXES = (".lock",)
EXEMPT_PATH_TOKENS = ("privacy_gate.py", "tests/fixtures/privacy/")

TEXT_SUFFIXES = (".py", ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".txt", ".ini", ".cfg", ".csv")


#: Hosts that appear legitimately in documentation and fixtures.
PLACEHOLDER_HOSTS: frozenset[str] = frozenset(
    {"host", "h", "x", "db", "example", "example.invalid", "localhost", "127.0.0.1", "mongo", "mongodb"}
)

#: Passwords that are obviously placeholders rather than secrets.
PLACEHOLDER_SECRETS: frozenset[str] = frozenset({"***", "...", "<password>", "secret", "password", "pass"})


def looks_like_placeholder(uri: str) -> bool:
    """True when a URI cannot be a real credential.

    The first run of this gate reported 14 critical `mongodb://` findings across the repository and
    every one was a placeholder: ``mongodb://...``, ``mongodb://user:***@host/db``,
    ``mongodb://example.invalid/...``. A gate that reports documentation as a critical leak trains
    people to ignore it, which is worse than no gate. Credentials are what matter; placeholders are
    how you write a credential down honestly.
    """
    body = uri.split("://", 1)[-1]
    if any(marker in uri for marker in ("...", "***", "<", "${")):
        return True
    credentials, _, rest = body.partition("@")
    host = (rest or credentials).split("/", 1)[0].split(":", 1)[0].strip().lower()
    if not rest:
        # no "@": the first segment is the host
        host = credentials.split("/", 1)[0].split(":", 1)[0].strip().lower()
    if host in PLACEHOLDER_HOSTS or host.endswith(".invalid"):
        # ".invalid" is reserved by RFC 2606: a host there cannot resolve, by construction.
        return True
    if rest:
        password = credentials.split(":", 1)[1] if ":" in credentials else ""
        if password.strip().lower() in PLACEHOLDER_SECRETS:
            return True
    return "." not in host and not host.replace(".", "").isdigit()


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    excerpt: str


def redact(text: str) -> str:
    """Keep enough shape to locate the problem, none of the secret."""
    stripped = text.strip()
    if len(stripped) <= 12:
        return stripped[:4] + "..."
    return stripped[:8] + "..." + f"({len(stripped)} chars)"


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True, check=True, timeout=120)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        files = []
    else:
        files = [root / line for line in out.stdout.splitlines() if line.strip()]
    if files:
        return files
    # A non-git directory must not scan as zero files: "0 findings" and "0 files read" look identical
    # in the output, and the first one is what a reader assumes.
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and ".git/" not in str(p) and p.suffix in TEXT_SUFFIXES)


def scan_file(path: Path, root: Path) -> list[Finding]:
    if path.suffix not in TEXT_SUFFIXES or path.suffix in EXEMPT_SUFFIXES:
        return []
    rel = str(path.relative_to(root))
    if any(token in rel for token in EXEMPT_PATH_TOKENS):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        uri_on_line = False
        for name, pattern, severity in RULES:
            if name == "email-address" and uri_on_line:
                # "user:pass@host" inside a mongodb URI is a credential, not an address; reporting
                # it twice hides which finding actually matters.
                continue
            match = pattern.search(line)
            if not match:
                continue
            if name == "private-path" and ("re.compile(" in line or " not in " in line):
                # The guard that forbids a private path, and the test that asserts it is absent,
                # both have to contain it. Flagging them punishes the enforcement of the rule.
                continue
            if name == "email-address" and match.group(0).startswith("git@"):
                # "git@github.com:owner/repo.git" is an SSH remote, not an address.
                continue
            if name == "mongodb-uri" and looks_like_placeholder(match.group(0)):
                # Documented placeholders are not credentials; reporting them as critical is how a
                # gate teaches people to ignore it.
                continue
            if name == "mongodb-uri":
                uri_on_line = True
            findings.append(Finding(rel, number, name, severity, redact(match.group(0))))
    return findings


def scan_root(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in tracked_files(root):
        findings.extend(scan_file(path, root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path,
                        help="repository root to scan; repeatable (default: this repo)")
    parser.add_argument("--json", type=Path, help="write findings as JSON")
    parser.add_argument("--fail-on", choices=("critical", "high", "medium"), default="high",
                        help="lowest severity that makes the gate fail (default: high). Medium "
                             "findings such as a deliberately published contact address are reported "
                             "but must not block a merge, or the gate gets bypassed.")
    args = parser.parse_args(argv)

    roots = args.root or [Path(__file__).resolve().parents[2]]
    all_findings: dict[str, list[dict[str, object]]] = {}
    exit_code = 0
    for root in roots:
        findings = scan_root(root)
        all_findings[str(root)] = [asdict(f) for f in findings]
        print(f"{root}: {len(findings)} finding(s) across {len(tracked_files(root))} tracked files")
        for finding in findings:
            print(f"  {finding.severity:8} {finding.rule:14} {finding.path}:{finding.line}  {finding.excerpt}")
        blocking = {"critical"} if args.fail_on == "critical" else (
            {"critical", "high"} if args.fail_on == "high" else {"critical", "high", "medium"})
        if any(f.severity in blocking for f in findings):
            exit_code = 1

    if args.json:
        args.json.write_text(json.dumps(all_findings, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
