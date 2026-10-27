"""Country x stage coverage audit for EP24 (#306, deliverable 2).

The question this answers is not "did the stage run" but "**is there a result**", and the difference
between those two is where this pipeline's real defects live. Three states are distinguished and must
never be collapsed:

* **not started** -- every column the stage is contracted to append is absent or empty;
* **failed** -- the stage's own status column says so (an explicit negative, not a missing value);
* **completed** -- at least one contracted output column carries something.

A fourth state is the one worth hunting: **status says success, output is empty**. That is a stage
claiming completion without a substantive result, and no denominator based on "status = ok" will ever
show it.

Counts only leave this module. The audit never emits row content, quotes, transcripts or identifiers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "FAILURE_STATUSES",
    "SUCCESS_STATUSES",
    "audit_stage",
    "audit_rows",
    "empty_value",
    "status_of",
]

#: Status values that mean the stage reported a real result.
SUCCESS_STATUSES: frozenset[str] = frozenset(
    {"ok", "done", "success", "succeeded", "complete", "completed", "available", "ready"}
)

#: Status values that mean the stage reported a failure. Anything else non-empty is "other", and is
#: reported as its own bucket rather than guessed at.
FAILURE_STATUSES: frozenset[str] = frozenset(
    {"error", "failed", "failure", "timeout", "cancelled", "canceled", "skipped", "unavailable"}
)


def empty_value(value: Any) -> bool:
    """True when a value carries no result.

    ``None``, empty/blank strings, and empty containers are all "no result". ``0`` and ``False`` are
    results: a stage that legitimately counts zero must not be reported as not-started.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def status_of(row: Mapping[str, Any], status_columns: Sequence[str]) -> str:
    """The stage's own status, lowercased. The first non-empty status column wins."""
    for column in status_columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return ""


def audit_stage(
    rows: Sequence[Mapping[str, Any]],
    *,
    number: int,
    name: str,
    appends: Sequence[str],
    status_columns: Sequence[str],
) -> dict[str, Any]:
    """Coverage for one stage over one country's canonical rows."""
    #: Columns that carry a result, excluding the status/reason bookkeeping columns. Otherwise a
    #: stage could look "completed" because it wrote its own failure reason.
    output_columns = tuple(
        column for column in appends
        if not column.endswith(("_status", "_reason", "_error", "_model", "_version", "_runtime_ms",
                                "_generated_at", "_prompt_version", "_context_sha256"))
    )
    if not output_columns:
        # A stage that contracts no output columns (e.g. the RDF stage writes elsewhere) cannot be
        # audited from this frame. Reporting 0% would be a fabricated finding.
        return {
            "stage": number, "name": name, "contracted_columns": len(appends),
            "present_columns": 0, "missing_columns": [], "rows": len(rows),
            "completed": 0, "failed": 0, "not_started": 0, "other_status": 0,
            "claimed_but_empty": 0, "coverage_pct": None, "auditable": False,
            "note": "stage contracts no output columns in this frame; audit it from its own store",
        }
    present_columns = [c for c in appends if any(c in row for row in rows)]

    not_started = completed = failed = claimed_but_empty = other_status = 0
    other_values: dict[str, int] = {}
    for row in rows:
        status = status_of(row, status_columns)
        has_output = any(not empty_value(row.get(column)) for column in output_columns)
        if has_output:
            completed += 1
            continue
        if status in SUCCESS_STATUSES:
            # The interesting defect: the stage says it succeeded and produced nothing.
            claimed_but_empty += 1
        elif status in FAILURE_STATUSES:
            failed += 1
        elif status:
            # e.g. REPROCESS / DELETE work-queue markers: honest non-results, not failures and not
            # successes. Folding them into either bucket would misstate the queue.
            other_status += 1
            other_values[status] = other_values.get(status, 0) + 1
        else:
            not_started += 1

    return {
        "stage": number,
        "name": name,
        "contracted_columns": len(appends),
        "present_columns": len(present_columns),
        "missing_columns": sorted(set(appends) - set(present_columns)),
        "rows": len(rows),
        "completed": completed,
        "failed": failed,
        "not_started": not_started,
        "other_status": other_status,
        "other_status_values": dict(sorted(other_values.items())),
        "claimed_but_empty": claimed_but_empty,
        "coverage_pct": round(100 * completed / len(rows), 1) if rows else 0.0,
        "auditable": True,
    }


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Iterable[Any],
    *,
    status_suffix: str = "_status",
) -> dict[str, Any]:
    """Audit every contracted stage over one country, plus the identity/language facts worth knowing.

    ``contract`` is any iterable of objects exposing ``number``, ``name`` and ``appends`` (the
    repository's own ``ep24_stage_contract.STAGE_CONTRACT`` satisfies this without importing it).
    """
    stages: list[dict[str, Any]] = []
    for stage in contract:
        statuses = tuple(c for c in stage.appends if c.endswith(status_suffix))
        stages.append(audit_stage(
            rows,
            number=stage.number,
            name=stage.name,
            appends=stage.appends,
            status_columns=statuses,
        ))

    languages: dict[str, int] = {}
    for row in rows:
        value = str(row.get("asr_language") or "").strip().lower()
        if value:
            languages[value] = languages.get(value, 0) + 1

    media = [str(row.get("allas_filename") or "").strip() for row in rows]
    media = [m for m in media if m]

    return {
        "rows": len(rows),
        "stages": stages,
        "asr_language": dict(sorted(languages.items())),
        "rows_with_media_key": len(media),
        "distinct_media": len(set(media)),
        "rows_sharing_a_media_key": len(media) - len(set(media)),
    }
