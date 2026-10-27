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
from dataclasses import dataclass, asdict
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
    ("private-path", re.compile(r"(?:/scratch/project_\d+|/mnt/workspace|/home/[a-z]+/(?:\.hermes|work))"), "high"),
    ("email-address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "medium"),
)

#: Paths whose matches are documentation of the rule itself, or fixtures that must
#: contain a pattern to test the gate.
EXEMPT_SUFFIXES = (".lock",)
EXEMPT_PATH_TOKENS = ("privacy_gate.py", "tests/fixtures/privacy/")

TEXT_SUFFIXES = (".py", ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".txt", ".ini", ".cfg", ".csv")


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
        return []
    return [root / line for line in out.stdout.splitlines() if line.strip()]


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
        for name, pattern, severity in RULES:
            match = pattern.search(line)
            if match:
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
        if findings:
            exit_code = 1

    if args.json:
        args.json.write_text(json.dumps(all_findings, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
