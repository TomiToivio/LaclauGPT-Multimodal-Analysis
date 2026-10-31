"""Non-blocking EP24 stage eligibility (#342).

The pipeline used to gate a stage on its predecessor's ``status == "complete"``. That is stricter than
the rules require and it is why a partial upstream stage stopped everything behind it: a record with a
perfectly usable transcript but no video never reached discourse, DNA, SNA or RDF.

Four things are kept separate here, because collapsing them is what produced the confusion in #333:

* **execution status** — what the stage did: not_run / running / succeeded / partial / failed / skipped
* **usable output** — whether the artifact a consumer needs is actually present
* **diagnostic** — why, in one line, without rewriting history
* **retry eligibility** — whether this record/stage should be picked up again

The rule that matters most: **a record with usable output is never retried**, even if an old failure
flag or an empty index cell says otherwise. Conversely an old failure flag does not make a stage
retryable when a valid result exists.

This module is storage-agnostic and pure: it reads row mappings, returns decisions, and never touches
Mongo, files or the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Execution status. Deliberately finer than the two-state success/failure the orchestrator had.
NOT_RUN = "not_run"
RUNNING = "running"
SUCCEEDED = "succeeded"
PARTIAL = "partial"
FAILED = "failed"
SKIPPED = "skipped"

#: Values that mean "a later stage may consume this", most permissive first.
USABLE_STATUSES = frozenset({SUCCEEDED, PARTIAL})

#: Fields that, when non-empty, prove a stage produced something a downstream consumer can read.
OUTPUT_FIELDS: dict[int, tuple[str, ...]] = {
    0: ("processing_status",),
    1: ("asr_transcript", "ocr_1", "frame_file"),
    2: ("frame_analysis_1", "frame_quality_status"),
    3: ("vllm_video_analysis", "vllm_structured_output", "vllm_video_markdown_analysis"),
    4: ("summary_analysis", "summary_summary_md"),
    5: ("postprocess_entities", "postprocess_themes", "postprocess_summary_md"),
    6: ("laclau_structured_json", "laclau_summary_md"),
    7: ("dna_statements_json", "dna_statement_count", "dna_analysis_markdown"),
    8: ("sna_nodes_json", "sna_edges_json"),
    9: ("rdf_ttl_path",),
}

#: Status field per stage, where the pipeline records one.
STATUS_FIELDS: dict[int, str] = {
    0: "processing_status",
    1: "preprocess_status",
    2: "frame_analysis_status",
    3: "vllm_structured_status",
    4: "summary_quality_status",
    5: "",
    6: "laclau_status",
    7: "",
    8: "",
    9: "",
}

SUCCESS_WORDS = frozenset({"ok", "done", "success", "complete", "completed", "available", "cached"})

#: The minimum a stage needs from its predecessor, expressed as *data*, not as a predecessor status.
#: Absence of one route is not absence of all routes: text-only records can reach discourse without
#: video, and DNA can be extracted from partial discourse.
INPUT_ROUTES: dict[int, tuple[tuple[str, ...], ...]] = {
    1: (("processing_status",),),
    2: (("frame_file", "frame_timestamp_seconds"), ("asr_transcript",)),
    3: (("frame_file", "frame_timestamp_seconds"),),
    4: (("asr_transcript",), ("ocr_1",), ("frame_analysis_1",)),
    5: (("summary_analysis", "summary_summary_md"), ("asr_transcript",)),
    6: (("postprocess_themes",), ("postprocess_entities",), ("summary_analysis",), ("asr_transcript",)),
    7: (("laclau_structured_json",), ("laclau_summary_md",), ("postprocess_themes",),
        ("asr_transcript",)),
    8: (("dna_statements_json", "dna_statement_count"), ("postprocess_entities",)),
    9: (("dna_statements_json",), ("sna_nodes_json",), ("laclau_structured_json",)),
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("[]", "{}", "null", "None") else text


def _status(row: Mapping[str, Any], step: int) -> str:
    field = STATUS_FIELDS.get(step) or ""
    return _text(row.get(field)).lower() if field else ""


def output_present(row: Mapping[str, Any], step: int) -> bool:
    """Whether the stage's consumer-facing artifact exists, independent of any status word."""
    fields = OUTPUT_FIELDS.get(step, ())
    return any(_text(row.get(field)) for field in fields)


def usable_input(row: Mapping[str, Any], step: int) -> tuple[bool, str]:
    """Whether the stage has a minimum valid input route, ignoring predecessor status.

    Returns ``(ok, reason)`` where ``reason`` names the route taken or why none exists. Checking data
    rather than status is the whole point: a predecessor marked ``partial`` may still have produced a
    perfectly good transcript.
    """
    routes = INPUT_ROUTES.get(step)
    if not routes:
        return True, "no prerequisite"
    for route in routes:
        if all(_text(row.get(field)) for field in route):
            return True, "input via " + "+".join(route)
    return False, "no usable input (looked for " + " | ".join("+".join(r) for r in routes) + ")"


@dataclass(frozen=True)
class Assessment:
    """One record's verdict for one stage."""

    stage: int
    execution_status: str
    output_available: bool
    diagnostic: str
    retry_eligible: bool
    reason_code: str = ""

    @property
    def usable_by_downstream(self) -> bool:
        """Downstream stages may proceed on this, with the diagnostic attached."""
        return self.execution_status in USABLE_STATUSES


def assess(row: Mapping[str, Any], step: int, *, retry_errors: bool = True,
           include_empty: bool = False) -> Assessment:
    """Classify one record for one stage.

    * usable output -> succeeded, never retryable
    * status says success but no output -> partial, retryable (the artifact is what consumers need)
    * status says failure with no output -> failed, retryable
    * no output, no status, no usable input -> skipped (recorded, not retried)
    * no output, no status, usable input -> not_run, retryable
    """
    has_output = output_present(row, step)
    status = _status(row, step)
    ok_input, input_note = usable_input(row, step)

    if has_output:
        return Assessment(step, SUCCEEDED, True, "", False, "output_present")

    if status and status not in SUCCESS_WORDS:
        # A recorded failure with no artifact. Retryable, because nothing usable exists.
        return Assessment(step, FAILED, False, f"status={status}", retry_errors, "failed_without_output")

    if status and status in SUCCESS_WORDS:
        # The stage claims success but the consumer-facing artifact is missing. Retrying is opt-in:
        # this is also the signature of a genuinely empty result, and rerunning those by default would
        # spend compute to rediscover that nothing was there.
        return Assessment(step, PARTIAL, False,
                          "status claims success but no output field is populated",
                          include_empty, "success_without_output")

    if not ok_input:
        return Assessment(step, SKIPPED, False, input_note, False, "no_usable_input")

    # Nothing recorded, usable input available: never run.
    return Assessment(step, NOT_RUN, False, input_note, True, "not_run")


def select_reprocess(
    rows: Iterable[Mapping[str, Any]],
    steps: Sequence[int],
    *,
    countries: Sequence[str] | None = None,
    include_empty: bool = False,
    retry_errors: bool = True,
    record_id_field: str = "record_id",
    country_field: str = "country",
) -> dict[str, Any]:
    """Build the manifest a selective rerun needs: what to run, what to skip, and why.

    Deterministic: sorted by record id then stage, so two runs produce the same manifest and a diff is
    meaningful.
    """
    wanted_countries = {c.lower() for c in countries} if countries else None
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row.get(record_id_field) or "")
        if not record_id:
            continue
        country = str(row.get(country_field) or "").lower()
        if wanted_countries is not None and country and country not in wanted_countries:
            continue
        for step in steps:
            verdict = assess(row, step, retry_errors=retry_errors, include_empty=include_empty)
            entry = {
                "record_id": record_id,
                "country": country,
                "stage": step,
                "reason_code": verdict.reason_code,
                "diagnostic": verdict.diagnostic,
            }
            if verdict.retry_eligible:
                selected.append(entry)
            else:
                skipped.append(entry)
    selected.sort(key=lambda item: (item["country"], item["record_id"], item["stage"]))
    skipped.sort(key=lambda item: (item["country"], item["record_id"], item["stage"]))
    return {
        "selected": selected,
        "skipped": skipped,
        "counts": {
            "selected": len(selected),
            "skipped": len(skipped),
            "by_stage": {str(step): sum(1 for e in selected if e["stage"] == step) for step in steps},
            "by_reason": _count_reasons(selected),
        },
    }


def _count_reasons(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["reason_code"]] = counts.get(entry["reason_code"], 0) + 1
    return dict(sorted(counts.items()))


def nonblocking_query(step: int, *, retry_errors: bool = True) -> dict[str, Any]:
    """A Mongo prefilter for a non-blocking stage run.

    Deliberately *wider* than the old ``predecessor.status == complete`` gate: it requires the data the
    stage actually consumes and leaves the precise decision to :func:`assess` on the fetched rows, where
    partial upstream evidence is visible. A prefilter cannot express "any of three input routes" cleanly
    without risking a false negative, so it only excludes rows that obviously cannot run.
    """
    query: dict[str, Any] = {}
    if step == 1:
        query["processing_status"] = {"$nin": ["DELETE", "delete"]}
    elif step == 2:
        query["$or"] = [
            {"frame_file": {"$exists": True, "$nin": ["", None]}},
            {"asr_transcript": {"$exists": True, "$nin": ["", None]}},
        ]
    elif step == 3:
        query["frame_file"] = {"$exists": True, "$nin": ["", None]}
        query["frame_timestamp_seconds"] = {"$exists": True, "$nin": ["", None]}
    elif step >= 6:
        # The old gate demanded processing_status OK for everything downstream. Keep that only where
        # the research design requires it (records marked DELETE are excluded), and let the row-level
        # assessment decide the rest.
        query["processing_status"] = {"$nin": ["DELETE", "delete"]}
    if not retry_errors:
        query["$and"] = [{"$or": [
            {f"step{step}_status.status": {"$exists": False}},
            {f"step{step}_status.status": {"$in": ["pending", "retry"]}},
        ]}]
    return query


def summary_line(manifest: Mapping[str, Any]) -> str:
    """One-line summary for logs and job output."""
    counts = manifest.get("counts", {})
    stages = ", ".join(f"{k}:{v}" for k, v in (counts.get("by_stage") or {}).items())
    return (f"selected {counts.get('selected', 0)} record-stage pairs, "
            f"skipped {counts.get('skipped', 0)} ({stages or 'no stages'})")
