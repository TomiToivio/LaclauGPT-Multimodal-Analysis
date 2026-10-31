"""Read-only EP24 FI/PL/PT export from shared MongoDB (#306, deliverable 1).

Design rules, in order of importance:

1. **Read-only.** Nothing here writes to MongoDB; the CLI opens the client with a read preference and
   only calls ``find``. A wrong export can be deleted and re-run; a wrong write cannot.
2. **Never invent an identity.** A row without ``allas_filename``/``_storage_id``/``video_id`` is
   quarantined and counted, never given a synthetic id. Absent results are not negative coding.
3. **One canonical row per unit.** The unit of analysis is explicit (``video`` keeps one row per
   media file, ``ingest`` keeps one per pipeline run) and the loser of a duplicate group is dropped
   deterministically — highest stage output wins, ties broken by storage id — so two runs of the
   exporter produce byte-identical output.
4. **Provenance travels with the data.** The manifest records which collections were read, how many
   documents each held, and the pipeline fields the source itself carries.

This module holds the pure logic so it can be tested without a database. The CLI is
``scripts/ep24/export_mongo_results.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "COUNTRY_SLUGS",
    "DEFAULT_RECORD_UNIT",
    "EXPORTER_VERSION",
    "RECORD_UNITS",
    "ExportError",
    "child_collection",
    "collection_name",
    "discover_provenance",
    "distinct_media",
    "media_key",
    "project_row",
    "quarantine_reason",
    "record_identity",
    "DEFAULT_TEXT_THRESHOLD",
    "select_canonical",
    "split_heavy_fields",
    "write_text_bundle",
    "stage_score",
    "write_manifest",
    "write_rows",
]

EXPORTER_VERSION = "ep24-mongo-export-1"

COUNTRY_SLUGS: tuple[str, ...] = ("finland", "poland", "portugal")

#: The unit of analysis. Same contract as the visualization exporters (#205): the row count is only
#: meaningful next to the unit, and ``video`` is the researcher-facing default.
RECORD_UNITS: tuple[str, ...] = ("video", "ingest")
DEFAULT_RECORD_UNIT = "video"

#: Country-level collections, by role. ``dataframe`` is the superset record: it carries every stage
#: field plus the identity columns, which is why it is the canonical source rather than ``analysis``.
CANONICAL_SUFFIX = "_dataframe"

#: Child collections joined to the canonical row by ``_storage_id``. Their absence is a finding, not
#: an error: Poland has no ``_dna_statements`` and Poland/Portugal have no ``_sna`` at all.
CHILD_SUFFIXES: dict[str, str] = {
    "entities": "_entities",
    "themes": "_themes",
    "dna_statements": "_dna_statements",
    "sna": "_sna",
}

#: Fields whose presence means a stage produced something. Counting these gives a deterministic
#: "most complete row" score without pretending to know the pipeline's internal contract.
STAGE_FIELDS: tuple[str, ...] = (
    "asr_transcript", "asr_translated", "asr_model",
    "ocr_1", "ocr_backend",
    "frame_analysis_1", "frame_analysis_status",
    "summary", "summary_quality", "video_summary",
    "discourse_analysis", "laclau_analysis",
    "dna_statements", "entities", "themes",
)


class ExportError(RuntimeError):
    """Raised when the export cannot be honest: missing config, unknown country, bad unit."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def collection_name(country: str, suffix: str = "") -> str:
    """``laclaugpt_ep2024_reprocess_<country><suffix>``."""
    if country not in COUNTRY_SLUGS:
        raise ExportError(f"unknown country {country!r}; expected one of {COUNTRY_SLUGS}")
    return f"laclaugpt_ep2024_reprocess_{country}{suffix}"


def child_collection(country: str, role: str) -> str:
    if role not in CHILD_SUFFIXES:
        raise ExportError(f"unknown child role {role!r}; expected one of {sorted(CHILD_SUFFIXES)}")
    return collection_name(country, CHILD_SUFFIXES[role])


def media_key(row: Mapping[str, Any]) -> str:
    """The media identity: the most stable key the row actually carries."""
    return (_text(row.get("allas_filename")) or _text(row.get("_storage_id"))
            or _text(row.get("video_id")))


def record_identity(row: Mapping[str, Any], unit: str = DEFAULT_RECORD_UNIT) -> str:
    """Namespaced identity for the chosen unit, or ``""`` when the row carries no usable id.

    ``video``  -- one record per media file (``allas_filename`` first).
    ``ingest`` -- one record per pipeline run (``_storage_id`` first).

    The prefix is part of the id so a ``video`` export and an ``ingest`` export can never be joined
    by accident: the same video appears in both, under different keys, on purpose.
    """
    if unit not in RECORD_UNITS:
        raise ExportError(f"unknown record unit {unit!r}; expected one of {RECORD_UNITS}")
    if unit == "ingest":
        native = _text(row.get("_storage_id")) or _text(row.get("video_id"))
        return f"ingest:{native}" if native else ""
    native = media_key(row)
    return f"video:{native}" if native else ""


def quarantine_reason(row: Mapping[str, Any], unit: str = DEFAULT_RECORD_UNIT) -> str | None:
    """Why a row cannot be exported, or ``None`` when it can. Never invents an id to avoid this."""
    if not record_identity(row, unit):
        return "no_identity"
    return None


def stage_score(row: Mapping[str, Any]) -> int:
    """How much stage output a row carries. Deterministic, and deliberately naive."""
    score = 0
    for name in STAGE_FIELDS:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, (str, bytes)) and not str(value).strip():
            continue
        if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
            continue
        score += 1
    return score


def select_canonical(
    rows: Iterable[Mapping[str, Any]], unit: str = DEFAULT_RECORD_UNIT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one row per identity and count everything that was dropped.

    Sorting is by ``(-stage_score, identity)`` so the winner never depends on cursor order.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    quarantined = 0
    for row in rows:
        if quarantine_reason(row, unit) is not None:
            quarantined += 1
            continue
        grouped.setdefault(record_identity(row, unit), []).append(row)

    kept: list[dict[str, Any]] = []
    collapsed = 0
    for identity in sorted(grouped):
        candidates = grouped[identity]
        collapsed += len(candidates) - 1
        best = min(candidates, key=lambda r: (-stage_score(r), _text(r.get("_storage_id")),
                                              _text(r.get("allas_filename"))))
        record = dict(best)
        record["record_id"] = identity
        record["stage_score"] = stage_score(best)
        kept.append(record)

    return kept, {
        "records": len(kept),
        "duplicates_collapsed": collapsed,
        "quarantined": quarantined,
        "distinct_media": distinct_media(kept),
    }


def distinct_media(rows: Iterable[Mapping[str, Any]]) -> int:
    return len({key for key in (media_key(r) for r in rows) if key})


def _csv_safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def project_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the Mongo ``_id`` (not reproducible across environments) and keep everything else."""
    return {k: _csv_safe(v) for k, v in row.items() if k != "_id"}


#: Text longer than this many characters is moved out of the row CSV into the text bundle. The
#: canonical frame is for filtering and counting; transcripts are for reading one record at a time.
DEFAULT_TEXT_THRESHOLD = 1000


#: Marker left in the index CSV where a heavy cell was moved to the text bundle (#331.3).
#: An elided cell that reads as empty is indistinguishable from an absent stage; this keeps coverage
#: legible in the index while the full value stays in ``<country>_text.jsonl.gz``.
HEAVY_SENTINEL = "<<elided: {chars} chars in {country}_text.jsonl.gz>>"


def split_heavy_fields(
    row: Mapping[str, Any],
    *,
    threshold: int = DEFAULT_TEXT_THRESHOLD,
    sentinel: str | None = None,
    country: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(lean_row, heavy_fields)``.

    Without this split the Finland frame alone is ~900 MB of transcript and context JSON, which no
    dashboard will load. The record survives in the lean CSV; its prose survives in the text bundle,
    keyed by the same ``record_id``.

    Passing ``sentinel`` leaves a readable marker in the lean row instead of deleting the cell, so a
    consumer can tell "elided, fetch the bundle" from "the stage produced nothing".
    """
    lean: dict[str, Any] = {}
    heavy: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > threshold:
            heavy[key] = value
            if sentinel:
                lean[key] = sentinel.format(chars=len(value), country=country or "country",
                                            field=key)
        else:
            lean[key] = value
    return lean, heavy


def heavy_field_report(rows: Sequence[Mapping[str, Any]], *,
                       threshold: int = DEFAULT_TEXT_THRESHOLD) -> dict[str, int]:
    """How many rows had each field elided, so elision becomes data rather than a silent hole."""
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and len(value) > threshold:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_text_bundle(path: Path, rows: Sequence[Mapping[str, Any]], *, threshold: int = DEFAULT_TEXT_THRESHOLD) -> int:
    """Write ``record_id`` + the heavy fields, gzipped JSONL. Returns the record count."""
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            _lean, heavy = split_heavy_fields(row, threshold=threshold)
            if not heavy:
                continue
            payload = {"record_id": row.get("record_id"), **heavy}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Write a CSV with the union of keys, deterministically ordered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    columns: list[str] = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_safe(row.get(k)) for k in columns})
    return len(rows)


def discover_provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the data says about itself: pipeline versions and the newest timestamps it carries.

    Reported, never guessed. A missing field is reported as absent rather than filled in.
    """
    pipelines: set[str] = set()
    timestamps: list[str] = []
    for row in rows:
        pipeline = row.get("_pipeline")
        if isinstance(pipeline, Mapping):
            for key in ("version", "run_id", "name"):
                value = _text(pipeline.get(key))
                if value:
                    pipelines.add(f"{key}={value}")
        elif _text(pipeline):
            pipelines.add(_text(pipeline))
        for key in ("preprocess_completed_at", "created_at", "updated_at", "processed_at"):
            value = _text(row.get(key))
            if value:
                timestamps.append(value)
    return {
        "pipeline_markers": sorted(pipelines),
        "newest_timestamp": max(timestamps) if timestamps else None,
        "timestamp_field_present": bool(timestamps),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: The visualization contract (#195): per step, the status field that decides failure and the
#: output fields that decide coverage. The dashboard's ``coverage_rows()`` reads
#: ``record_counts[country].exported`` as the denominator and ``stage_coverage[country][step]`` as
#: the numerator, so an export without these keys cannot be opened by the researcher dashboard.
COVERAGE_STEPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "0_ingest": ("processing_status", ("processing_status_reason", "needs_resplit")),
    "1_preprocess": ("preprocess_status", ("asr_transcript", "asr_translated", "ocr_1",
                                           "frame_file", "preprocess_completed_at")),
    "2_frame": ("frame_analysis_status", ("frame_analysis_1", "frame_quality_status")),
    "3_video": ("vllm_video_status", ("vllm_video_analysis", "vllm_video_markdown_analysis",
                                      "vllm_video_structured_json")),
    "4_summary": ("summary_quality_status", ("summary_analysis", "summary_summary_md",
                                             "summary_quality_reason")),
    "5_postprocess": ("", ("postprocess_entities", "postprocess_themes",
                           "postprocess_summary_md")),
    "6_discourse": ("laclau_status", ("laclau_structured_json", "laclau_summary_md")),
    "7_dna": ("", ("dna_statements_json", "dna_analysis_markdown")),
    "8_sna": ("", ("sna_nodes_json", "sna_edges_json")),
    "9_rdf": ("", ("rdf_ttl_path",)),
}

SUCCESS_STATUSES = frozenset({"done", "ok", "success", "complete", "completed", "available"})


#: Steps whose real output is a normalized child table, not a row-level cell (#332). A child-table
#: export leaves the row cell empty by design, so row-only coverage reports the stage as unavailable.
CHILD_STEPS: dict[str, str] = {
    "7_dna": "dna_statements",
    "8_sna": "sna",
}


def stage_coverage(
    rows: Sequence[Mapping[str, Any]],
    child_counts: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per step: how many rows carry output, how many carry a failure, and whether it is measurable.

    ``available`` is false when no row carries any of the step's output fields: the source cannot
    express that step. That is a different finding from ``records: 0``, which means the step is
    present and produced nothing, and collapsing the two turns an absent stage into a null result.
    """
    coverage: dict[str, dict[str, Any]] = {}
    for step, (status_field, outputs) in COVERAGE_STEPS.items():
        measurable = any(any(_text(row.get(field)) for field in outputs) for row in rows) or any(
            field in row for row in rows for field in outputs)
        present = sum(1 for row in rows if any(_text(row.get(field)) for field in outputs))
        failed = 0
        if status_field:
            failed = sum(1 for row in rows
                         if _text(row.get(status_field)).lower()
                         and _text(row.get(status_field)).lower() not in SUCCESS_STATUSES)
        source = "rows" if present else "none"
        # A step that lives in a child table is measured by that table when the row cell is empty.
        # Reporting it from rows alone is what made DNA read as "unavailable" while the manifest said
        # 1,583 statements (#332): two files, two answers, same export.
        child_role = CHILD_STEPS.get(step)
        if child_role and child_counts is not None:
            child_total = int(child_counts.get(child_role, 0) or 0)
            if present == 0 and child_total > 0:
                present = child_total
                measurable = True
                source = "children"
            elif child_total > 0:
                source = "rows+children"
        coverage[step] = {
            "records": present,
            "available": measurable,
            "failed_or_other": failed,
            "status_field": status_field or None,
            "pct_of_exported": round(100 * present / len(rows), 1) if rows else 0.0,
            "source": source,
        }
    return coverage


def dashboard_contract(
    country_stats: Mapping[str, Mapping[str, Any]],
    rows_by_country: Mapping[str, Sequence[Mapping[str, Any]]],
    child_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """The #195 keys, built from the counts this exporter already produces.

    Additive on top of the exporter's own manifest: ``record_counts`` renames the exporter's
    ``duplicates_collapsed``/``quarantined`` to ``duplicates``/``invalid_ids`` and adds ``read``,
    because those are the names the dashboard and the coverage report read.
    """
    counts: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for country, stats in country_stats.items():
        rows = rows_by_country.get(country, ())
        counts[country] = {
            "read": int(stats.get("source_documents", stats.get("read", 0)) or 0),
            "exported": int(stats.get("records", 0)),
            "duplicates": int(stats.get("duplicates_collapsed", 0) or 0),
            "invalid_ids": int(stats.get("quarantined", 0) or 0),
            "distinct_media": int(stats.get("distinct_media", 0) or 0),
        }
        steps = stage_coverage(rows, (child_counts or {}).get(country))
        coverage[country] = steps
        inputs[country] = {
            "profile": "mongo",
            "stage_unavailable": sorted(step for step, entry in steps.items()
                                        if not entry["available"]),
        }
    counts["exported_total"] = sum(entry["exported"] for entry in counts.values()
                                   if isinstance(entry, Mapping))
    return {"record_counts": counts, "stage_coverage": coverage, "inputs": inputs}


def parent_key(row: Mapping[str, Any]) -> str:
    """The child row's parent key, without inventing one.

    ``parent_record_id`` is written by the exporter as an explicit alias of ``source_record_id``; the
    older field is still read so exports produced before #330 remain joinable. ``_storage_id`` is
    deliberately **not** a candidate: on child rows it holds the statement's own id, and joining on it
    silently returns nothing (#331.2).
    """
    for field in ("parent_record_id", "source_record_id"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def children_qa(
    child_rows: Sequence[Mapping[str, Any]],
    parent_ids: set[str],
    *,
    parent_field: str = "parent_record_id",
) -> dict[str, Any]:
    """Check that a child table can actually be joined to the exported corpus (#330).

    Reports rather than repairs: an orphan statement is counted, never given an invented parent, and a
    duplicate statement id is counted because a duplicated statement inflates a consumer's totals while
    looking like extra evidence.
    """
    rows = len(child_rows)
    statement_ids: list[str] = [str(row.get("statement_id") or "") for row in child_rows]
    non_empty = [value for value in statement_ids if value]
    unique_statements = len(set(non_empty))
    duplicate_statements = len(non_empty) - unique_statements
    resolved = 0
    orphans = 0
    missing_parent_field = 0
    for row in child_rows:
        parent = parent_key(row)
        if not parent:
            missing_parent_field += 1
            continue
        if parent in parent_ids:
            resolved += 1
        else:
            orphans += 1
    return {
        "rows": rows,
        "unique_statement_ids": unique_statements,
        "duplicate_statement_ids": duplicate_statements,
        "parent_field": parent_field,
        "parents_resolved": resolved,
        "orphans": orphans,
        "missing_parent_field": missing_parent_field,
        "join_complete": rows > 0 and orphans == 0 and missing_parent_field == 0,
    }


def write_manifest(
    output: Path,
    *,
    record_unit: str,
    children_unit: str = "ingest",
    heavy_fields: Mapping[str, Mapping[str, int]] | None = None,
    children_qa_report: Mapping[str, Mapping[str, Any]] | None = None,
    countries: Mapping[str, Mapping[str, Any]],
    child_counts: Mapping[str, Mapping[str, int]],
    collections_read: Mapping[str, Mapping[str, str]],
    provenance: Mapping[str, Mapping[str, Any]],
    outputs: Sequence[Path],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest is the contract: unit, counts, provenance and what was missing."""
    total_records = sum(int(c.get("records", 0)) for c in countries.values())
    total_media = sum(int(c.get("distinct_media", 0)) for c in countries.values())
    manifest = {
        "exporter_version": EXPORTER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "read_only": True,
        "record_unit": record_unit,
        "records": total_records,
        "distinct_media": total_media,
        "countries": dict(countries),
        "children": dict(child_counts),
        # The children block is exported from the pipeline's own collections, which are keyed at ingest
        # level; the record_counts block is keyed at ``record_unit``. Saying so is what lets a consumer
        # reconcile them instead of measuring two populations unknowingly (#331.1).
        "children_unit": children_unit,
        "heavy_fields": {country: dict(fields) for country, fields in (heavy_fields or {}).items()},
        "children_qa": {country: dict(report) for country, report in (children_qa_report or {}).items()},
        "collections_read": dict(collections_read),
        "provenance": dict(provenance),
        "outputs": [
            {
                "name": p.name,
                "bytes": p.stat().st_size if p.exists() else 0,
                "sha256": sha256_file(p) if p.exists() else None,
            }
            for p in outputs
        ],
    }
    manifest.update(contract or {})
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def config_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Mongo URI + database from the environment. The values are never logged or echoed."""
    env = os.environ if environ is None else environ
    uri = (env.get("LACLAUGPT_MONGODB_URI") or "").strip()
    database = (env.get("LACLAUGPT_MONGODB_DATABASE") or "").strip()
    if not uri or not database:
        raise ExportError(
            "set LACLAUGPT_MONGODB_URI and LACLAUGPT_MONGODB_DATABASE in the private environment"
        )
    return uri, database
