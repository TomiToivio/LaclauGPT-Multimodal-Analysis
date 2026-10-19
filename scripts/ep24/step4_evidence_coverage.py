"""Step 4 evidence-coverage worklist for the rerun gate (#236).

The runbook (docs/EP24_PROMPT_EVIDENCE_RUNBOOK.md) defines the gate as a *manual* review: pick 10
source IDs each for Finland, Poland and Portugal, run the stages on those only, and score evidence
fidelity by hand before any wider run. That review needs the runtime — the corpus, the model and
credentials — so it cannot be automated and should not be. What can be automated is everything up
to the point where a human has to look:

* select the 30 source IDs **deterministically**, so the "same source IDs" comparison in step 4
  compares like with like and a later run can reproduce the exact set;
* report, per selected record, how much of that record's own text actually reaches the model under
  the bounded packet, and which fields do not. That is the machine-checkable half of step 3:
  whether #244's budget quietly dropped source evidence the older cumulative prompt carried.

Read-only, offline, no model calls, no database writes. It never touches MongoDB or the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ep24_prompt_contract import build_stage_packet, clean, estimate_tokens  # noqa: E402

COUNTRY_CODES = {"finland": "FI", "poland": "PL", "portugal": "PT"}
# Fields that are machinery rather than source evidence, and must not be scored as omissions.
NON_EVIDENCE_PREFIXES = ("_", "raw_prompt", "summary_analysis", "vllm_structured_output")
MIN_FIELD_CHARS = 40


@dataclass(frozen=True)
class RecordReport:
    """What one record contains versus what the bounded packet exposes to the model."""

    source_id: str
    country: str
    row_chars: int
    packet_chars: int
    packet_tokens: int
    surfaced_fields: tuple[str, ...]
    omitted_fields: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return round(self.packet_chars / self.row_chars, 3) if self.row_chars else 1.0


def country_code(row: dict) -> str:
    """Normalize the country field, which appears as a name or an ISO code."""
    raw = clean(row.get("country")).casefold()
    return COUNTRY_CODES.get(raw, raw.upper())


def select_sample(rows: Sequence[dict], per_country: int = 10) -> list[dict]:
    """Pick a deterministic sample per country, so the same IDs recur on the next run.

    Sorted by source identity and truncated, rather than taken in file order: file order can change
    when upstream collections rerun, which would silently move the benchmark's baseline.
    """
    chosen: list[dict] = []
    by_country: dict[str, list[dict]] = {}
    for row in rows:
        by_country.setdefault(country_code(row), []).append(row)
    for code in sorted(by_country):
        ordered = sorted(by_country[code], key=lambda r: str(r.get("video_id") or r.get("id") or ""))
        chosen.extend(ordered[:per_country])
    return chosen


def report_record(row: dict) -> RecordReport:
    """Compare one record's own text with the bounded packet the model would receive."""
    packet = build_stage_packet(4, row).text
    text_fields = {
        key: text
        for key in row
        if len(text := clean(row.get(key))) >= MIN_FIELD_CHARS
        and not key.casefold().startswith(NON_EVIDENCE_PREFIXES)
    }
    surfaced = tuple(sorted(k for k, v in text_fields.items() if v[:200] in packet))
    return RecordReport(
        source_id=str(row.get("video_id") or row.get("id") or "<unknown>"),
        country=country_code(row),
        row_chars=sum(len(v) for v in text_fields.values()),
        packet_chars=len(packet),
        packet_tokens=estimate_tokens(packet),
        surfaced_fields=surfaced,
        omitted_fields=tuple(sorted(set(text_fields) - set(surfaced))),
    )


def sample_fingerprint(reports: Iterable[RecordReport]) -> str:
    """Stable identifier for the selected set, to quote in the review notes and the run log."""
    payload = "\n".join(sorted(f"{r.country}:{r.source_id}" for r in reports))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render(reports: Sequence[RecordReport]) -> str:
    """Worklist for the manual scoring: one line per record, plus the per-country tally."""
    lines = [
        "# EP24 Step 4 evidence-coverage worklist",
        "",
        f"sample fingerprint: `{sample_fingerprint(reports)}`  ({len(reports)} records)",
        "",
        "Score each record by hand against the source before any wider rerun "
        "(runbook step 3). `coverage` is packet chars / record text chars: a low value is the "
        "question to answer, not automatically a defect.",
        "",
        "| country | source_id | record chars | packet chars | tokens~ | coverage | omitted fields |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in sorted(reports, key=lambda r: (r.country, r.source_id)):
        omitted = ", ".join(report.omitted_fields) or "-"
        lines.append(
            f"| {report.country} | {report.source_id} | {report.row_chars} | "
            f"{report.packet_chars} | {report.packet_tokens} | {report.coverage} | {omitted} |"
        )
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.country] = counts.get(report.country, 0) + 1
    lines += ["", "per country: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    return "\n".join(lines) + "\n"


def read_rows(path: Path) -> list[dict]:
    """CSV in, dicts out. Deliberately the only input format: no database handle, no writes."""
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 4 evidence-coverage worklist for the rerun gate.")
    parser.add_argument("csv", type=Path, help="Step 4 input CSV (read-only)")
    parser.add_argument("--per-country", type=int, default=10, help="records per country (default 10)")
    parser.add_argument("--json", type=Path, help="also write the machine-readable report here")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    selected = select_sample(rows, per_country=args.per_country)
    if not selected:
        print(f"no records selected from {args.csv}", file=sys.stderr)
        return 2
    reports = [report_record(row) for row in selected]
    print(render(reports), end="")
    if args.json:
        args.json.write_text(
            json.dumps([r.__dict__ | {"coverage": r.coverage} for r in reports], indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
