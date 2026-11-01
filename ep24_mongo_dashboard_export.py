#!/usr/bin/env python3
"""Read-only EP24 (FI/PL/PT) export from shared MongoDB — issue #306, section 1.

Why this exists: the visualization exporter (#195) reads *CSV frames*, so someone had to turn the
reanalysis results in shared MongoDB into those frames first. That step lived in notebooks. This is
the documented, repeatable replacement:

    python ep24_mongo_dashboard_export.py --countries fi pl pt --output /private/ep24/mongo-export

Guarantees, each with a test:

* **Read-only.** Collections are opened with ``find``/``aggregate`` only; there is no write path.
* **No invented identifiers.** A document without a usable ``video_id`` *and* a media key is
  quarantined and counted, never given a synthetic key and never dropped silently.
* **One row per video.** Identity is the country-scoped media key, so several reanalysis runs of one
  video collapse to the row with the most stage output — the same rule the #195 exporter applies, so
  the two agree on denominators.
* **Missingness is not a negative finding.** A step with no source column reads
  ``available: false`` with a reason; only a step that ran and produced nothing is a zero.
* **No secrets in output.** The URI is never written; the manifest records only which environment
  variable supplied it.

The artifact contract is the #195 one: ``stage_coverage[country][step] = {records, available,
pct_of_exported}``, ``record_counts[country] = {read, exported, duplicates, invalid_ids,
distinct_media}``, ``inputs[country].stage_unavailable``. A dashboard built on that contract can read
this export without a second schema.

Collection names come from ``roihu_storage.collection_prefix`` rather than a hard-coded string, so a
dataset rename fails loudly instead of exporting nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from roihu_storage import StorageConfig

EXPORT_VERSION = "ep24-mongo-export/1.0"
DATASET = "ep2024_reprocess"

#: Short country keys researchers type, mapped to the slug used in collection names.
COUNTRIES: dict[str, str] = {"fi": "finland", "pl": "poland", "pt": "portugal"}

#: Child collections to read, and the table each becomes.
CHILD_PURPOSES: dict[str, str] = {
    "dna_statements": "dna_statements",
    "sna": "sna_nodes",
    "entities": "entities",
    "themes": "themes",
}

#: Field groups per pipeline step, copied from the #195 export contract
#: (``LaclauGPT-Data-Visualization: scripts/ep24/export_streaming.py``) so the two exporters describe
#: a record with the same vocabulary. ``status`` decides failure; ``human``/``machine`` are outputs.
STAGES: dict[str, dict[str, tuple[str, ...]]] = {
    "0_ingest": {"status": ("processing_status",),
                 "human": ("processing_status_reason", "needs_resplit"), "machine": ()},
    "1_preprocess": {"status": ("preprocess_status",),
                     "human": ("preprocess_note", "preprocess_completed_at", "asr_transcript",
                               "asr_translated", "ocr_1", "frame_file"),
                     "machine": ("asr_language", "asr_backend", "asr_model", "ocr_backend",
                                 "ocr_model", "frame_timestamp_seconds")},
    "2_frame": {"status": ("frame_analysis_status",),
                "human": ("frame_analysis_1",), "machine": ("frame_quality_status",)},
    "3_video": {"status": ("vllm_video_status",),
                "human": ("vllm_video_analysis", "vllm_video_markdown_analysis"),
                "machine": ("vllm_video_structured_json", "vllm_video_model")},
    "4_summary": {"status": ("summary_quality_status",),
                  "human": ("summary_analysis", "summary_summary_md"),
                  "machine": ("summary_quality_reason",)},
    "5_postprocess": {"status": (),
                      "human": ("postprocess_entities", "postprocess_themes",
                                "postprocess_summary_md"), "machine": ()},
    "6_discourse": {"status": ("laclau_status",),
                    "human": ("laclau_structured_json", "laclau_summary_md"),
                    "machine": ("laclau_abstention_reason",)},
    "7_dna": {"status": ("dna_status",), "human": ("dna_statements_json", "dna_analysis_markdown"),
              "machine": ()},
    "8_sna": {"status": (), "human": ("sna_nodes_json", "sna_edges_json"),
              "machine": ("sna_analysis_markdown",)},
    "9_rdf": {"status": (), "human": ("rdf_ttl_path",), "machine": ()},
}

SUCCESS_STATUSES = frozenset({"done", "ok", "success", "complete", "completed", "available"})
EMPTY_CELLS = frozenset({"", "nan", "none", "null", "na", "n/a", "-", "[]", "{}"})

IDENTITY_FIELDS = ("video_id", "_storage_id", "allas_filename")
PROVENANCE_FIELDS = ("source_type", "source_recording", "sequence_number", "video_filename",
                     "author_username", "account_type", "political_preference", "video_duration",
                     "video_duration_seconds")


class ExportError(RuntimeError):
    """A condition the export refuses to paper over."""


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #


def connection_settings(environ: Mapping[str, str] | None = None) -> tuple[str, str, str]:
    """(uri, database, uri_source) from the environment.

    Accepts both spellings the repositories use (``LACLAUGPT_MONGO_URI`` in the producer,
    ``LACLAUGPT_MONGODB_URI`` in the private runtime config). Returns the *name* of the variable,
    never the URI: the manifest is written to disk and often pasted into an issue.
    """
    env = os.environ if environ is None else environ
    for name in ("LACLAUGPT_MONGO_URI", "LACLAUGPT_MONGODB_URI", "MONGO_URI"):
        if env.get(name):
            database = (env.get("LACLAUGPT_MONGO_DATABASE")
                        or env.get("LACLAUGPT_MONGODB_DATABASE")
                        or unquote(urlsplit(env[name]).path.lstrip("/")) or "laclaugpt")
            return env[name], database, name
    raise ExportError("no MongoDB URI in the environment "
                      "(LACLAUGPT_MONGO_URI or LACLAUGPT_MONGODB_URI)")


def collection_name(country_slug: str, purpose: str, *, dataset: str = DATASET) -> str:
    """The real collection name, derived from the producer's own naming contract.

    ``dataset`` selects the run: the prefix is data, not code. The reprocessing run wrote
    ``laclaugpt_ep2024_reprocess_*``; the Monday500 refresh writes a run-scoped prefix such as
    ``laclaugpt_ep24_monday500_20261031_*``. Hard-coding one of them exports the wrong run's records
    when a newer run lands beside it, so callers that know the run pass it explicitly.
    """
    return StorageConfig(dataset=dataset, country=country_slug).collection(purpose)


def connect(uri: str, database: str, *, timeout_ms: int = 15000) -> Any:
    """A read-only client. Imported lazily so CSV-only workflows need no pymongo."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ExportError("pymongo is required for the Mongo export") from exc
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms,
                         readPreference="secondaryPreferred")
    client.admin.command("ping")
    return client[database]


def documents(collection: Any, query: Mapping[str, Any] | None = None, *,
              projection: Mapping[str, Any] | None = None,
              batch_size: int = 500, limit: int | None = None,
              retries: int = 2) -> Iterator[dict[str, Any]]:
    """Stream documents with a bounded batch and a retry on a transient read error.

    ``query`` filters, ``projection`` selects fields: they are separate arguments because passing a
    projection as the filter matches nothing at all and produces a silently empty export.
    """
    attempt = 0
    emitted = False
    while True:
        try:
            for index, doc in enumerate(
                    collection.find(query or {}, projection, batch_size=batch_size)):
                if limit is not None and index >= limit:
                    return
                emitted = True
                yield dict(doc)
            return
        except ExportError:
            raise
        except Exception as exc:  # pragma: no cover - network shape varies
            attempt += 1
            if emitted or attempt > retries:
                raise ExportError(f"reading {collection.name} failed: {type(exc).__name__}") from exc
            time.sleep(2 * attempt)


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def text(value: Any) -> str:
    """A scalar as trimmed text; a JSON-ish container becomes compact JSON."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    raw = str(value).strip()
    return "" if raw.lower() in EMPTY_CELLS else raw


def has_usable_identity(doc: Mapping[str, Any]) -> bool:
    """A record needs a video id *and* something that identifies its media."""
    return bool(text(doc.get("video_id")) and
                (text(doc.get("allas_filename")) or text(doc.get("_storage_id"))))


def record_key(doc: Mapping[str, Any], country: str) -> str:
    """Country-scoped media identity; '' when the document cannot be identified."""
    if not has_usable_identity(doc):
        return ""
    media = text(doc.get("allas_filename")) or text(doc.get("_storage_id"))
    parts = (country, media, text(doc.get("video_id")), text(doc.get("source_recording")),
             text(doc.get("sequence_number")), text(doc.get("source_type")))
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{country}/{text(doc.get('video_id'))}#{digest}"


def stage_fields() -> tuple[str, ...]:
    """Every field a row must carry: status, human and machine groups of every step."""
    return tuple(dict.fromkeys(field for spec in STAGES.values()
                               for group in spec.values() for field in group))


def projection() -> dict[str, int]:
    """The server-side projection: only the fields the artifact set carries.

    A cumulative document has ~126 keys including prompts, raw model responses and backend
    telemetry. Pulling all of them costs minutes per country and none of it reaches the export, so
    the projection is derived from the contract rather than guessed.
    """
    return {field: 1 for field in (*IDENTITY_FIELDS, *PROVENANCE_FIELDS, *stage_fields())}


def completeness(doc: Mapping[str, Any]) -> int:
    """How much pipeline output the document actually carries."""
    return sum(1 for spec in STAGES.values()
               if any(text(doc.get(field)) for field in (*spec["human"], *spec["machine"])))


def canonical_rows(docs: Iterable[Mapping[str, Any]],
                   country: str) -> tuple[list[dict], dict, set[str]]:
    """One row per video: most complete wins; unusable rows are counted, never invented.

    Also returns the union of source column names, because "this step has no source column" and
    "this step ran and produced nothing" are different findings and only the source can tell them
    apart once every row has been normalised to the full field set.
    """
    keep: dict[str, dict[str, Any]] = {}
    scores: dict[str, int] = {}
    seen = invalid = duplicates = 0
    columns: set[str] = set()
    carried = (IDENTITY_FIELDS, PROVENANCE_FIELDS, stage_fields())
    for doc in docs:
        seen += 1
        columns.update(doc.keys())
        key = record_key(doc, country)
        if not key:
            invalid += 1
            continue
        score = completeness(doc)
        if key in keep:
            duplicates += 1
            if score <= scores[key]:
                continue
        row = {field: text(doc.get(field)) for group in carried for field in group}
        row.update({"record_key": key, "country": country})
        keep[key] = row
        scores[key] = score
    return ([keep[key] for key in sorted(keep)],
            {"read": seen, "exported": len(keep), "duplicates": duplicates,
             "invalid_ids": invalid},
            columns)


def stage_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per step: how many records carry it, and whether it can be measured at all."""
    coverage: dict[str, Any] = {}
    for step, spec in STAGES.items():
        outputs = (*spec["human"], *spec["machine"])
        present = sum(1 for row in rows if any(text(row.get(field)) for field in outputs))
        failed = 0
        if spec["status"]:
            failed = sum(1 for row in rows
                         for field in spec["status"]
                         if text(row.get(field))
                         and text(row.get(field)).lower() not in SUCCESS_STATUSES)
        coverage[step] = {
            "records": present,
            "available": bool(rows) or present > 0,
            "failed_or_other": failed,
            "pct_of_exported": round(100 * present / len(rows), 1) if rows else 0.0,
            "status_field": spec["status"][0] if spec["status"] else None,
        }
    return coverage


def child_rows(docs: Iterable[Mapping[str, Any]], *, country: str) -> list[dict[str, str]]:
    """Flatten a child collection into one table, keeping its own provenance columns."""
    rows: list[dict[str, str]] = []
    for index, doc in enumerate(docs):
        row = {key: text(value) for key, value in doc.items() if key != "_id"}
        row.setdefault("child_index", str(index))
        row["country"] = country
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text(row.get(field)) for field in fields})
    return len(rows)


def write_data_dictionary(output: Path, manifest: Mapping[str, Any]) -> None:
    lines = ["# EP24 MongoDB export — data dictionary", "",
             f"Contract: `{EXPORT_VERSION}` (the #195 artifact contract)", "",
             "## Identity", "",
             "`record_key` = `country/<video_id>#<digest over media, video_id, recording, "
             "segment, source_type>`.",
             "One row per video: several reanalysis runs of one video collapse to the row carrying",
             "the most stage output, and the collapse is counted as `duplicates`.",
             "A document without a `video_id` **and** a media key (`allas_filename` or",
             "`_storage_id`) is quarantined and counted as `invalid_ids`; no id is fabricated.", "",
             "## Coverage", "",
             "`available: false` means this source cannot carry the step — it is not `records: 0`,",
             "which would read as \"the model ran and found nothing\".", "",
             "| country | step | records | exported | failed_or_other | available |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    counts = manifest.get("record_counts", {})
    for country, stages in (manifest.get("stage_coverage") or {}).items():
        exported = (counts.get(country) or {}).get("exported", 0)
        for step, entry in stages.items():
            lines.append(f"| {country} | {step} | {entry.get('records', 0)} | {exported} | "
                         f"{entry.get('failed_or_other', 0)} | {entry.get('available')} |")
    (output / "DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_countries(client: Any, countries: Sequence[str], output: Path, *,
                     limit: int | None = None, batch_size: int = 500,
                     query: Mapping[str, Any] | None = None, project: bool = True,
                     uri_source: str = "", dataset: str | None = None) -> dict[str, Any]:
    run_dataset = dataset or DATASET
    output.mkdir(parents=True, exist_ok=True)
    present = set(client.list_collection_names())
    manifest: dict[str, Any] = {
        "contract": EXPORT_VERSION,
        "builder_version": EXPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": run_dataset,
        "database": getattr(client, "name", ""),
        "uri_source": uri_source,
        "record_unit": "video",
        "record_counts": {},
        "stage_coverage": {},
        "inputs": {},
        "children": {},
    }
    combined: list[dict[str, Any]] = []
    children: dict[str, list[dict[str, str]]] = {table: [] for table in CHILD_PURPOSES.values()}

    for short in countries:
        slug = COUNTRIES[short]
        frame = collection_name(slug, "dataframe", dataset=run_dataset)
        manifest["inputs"][short] = {"country": slug, "profile": "mongo", "source": frame,
                                     "stage_unavailable": []}
        if frame not in present:
            manifest["inputs"][short].update(available=False,
                                             reason="collection not present in this database")
            manifest["record_counts"][short] = {"read": 0, "exported": 0, "duplicates": 0,
                                                "invalid_ids": 0, "distinct_media": 0}
            manifest["stage_coverage"][short] = {
                step: {"records": 0, "available": False, "failed_or_other": 0,
                       "pct_of_exported": 0.0,
                       "reason": "no collection for this country"} for step in STAGES}
            continue

        rows, stats, columns = canonical_rows(
            documents(client[frame], query, projection=projection() if project else None,
                      limit=limit, batch_size=batch_size), short)
        # A step whose defining columns are absent from every exported row cannot be measured: say
        # so instead of reporting the zero that a reader would interpret as "ran, found nothing".
        coverage = stage_coverage(rows)
        for step, spec in STAGES.items():
            outputs = (*spec["human"], *spec["machine"])
            if not any(field in columns for field in outputs):
                manifest["inputs"][short]["stage_unavailable"].append(step)
                coverage[step].update(records=0, available=False,
                                      reason="no such field in this source")
        stats["distinct_media"] = len({row.get("allas_filename") or row.get("_storage_id")
                                       for row in rows if row.get("allas_filename")
                                       or row.get("_storage_id")})
        manifest["record_counts"][short] = stats
        manifest["stage_coverage"][short] = coverage
        _write_csv(output / f"{short}.csv", rows)
        combined.extend(rows)

        for purpose, table in CHILD_PURPOSES.items():
            name = collection_name(slug, purpose, dataset=run_dataset)
            key = f"{short}_{table}"
            if name not in present:
                manifest["children"][key] = {"available": False, "records": 0,
                                             "reason": "collection not present"}
                continue
            if query or limit is not None:
                # Child schemas do not guarantee the parent's run/identity keys.
                # Fail closed instead of exporting unrelated rows under a subset manifest.
                manifest["children"][key] = {"available": False, "records": 0,
                    "reason": "subset export: child run/parent alignment is unverified"}
                continue
            kids = child_rows(documents(client[name], limit=limit, batch_size=batch_size),
                              country=short)
            manifest["children"][key] = {"available": bool(kids), "records": len(kids),
                                         "source": name}
            children[table].extend(kids)

    combined.sort(key=lambda row: row["record_key"])
    _write_csv(output / "combined.csv", combined)
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for table, rows in children.items():
        _write_csv(output / f"{table}.csv", rows)
        manifest["children"][table] = {"records": len(rows)}

    manifest["records"] = len(combined)
    manifest["distinct_media"] = len({row.get("allas_filename") or row.get("_storage_id")
                                      for row in combined
                                      if row.get("allas_filename") or row.get("_storage_id")})
    manifest["record_counts"]["exported_total"] = len(combined)
    write_data_dictionary(output, manifest)
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only EP24 FI/PL/PT export from shared MongoDB (issue #306).")
    parser.add_argument("--countries", nargs="+", default=list(COUNTRIES),
                        choices=list(COUNTRIES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--database", default=None, help="override the database name")
    parser.add_argument("--limit", type=int, default=None,
                        help="read at most N documents per collection (development subset)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--query", default="", metavar="JSON",
                        help="optional Mongo filter, for example one run id")
    parser.add_argument("--dataset", default=None,
                        help="dataset/run whose collections are read (collection prefix component); "
                             "defaults to the historical ep2024_reprocess run")
    parser.add_argument("--no-projection", dest="project", action="store_false", default=True,
                        help="fetch whole documents instead of only the contract's fields "
                             "(slower; useful when auditing what the source actually holds)")
    args = parser.parse_args(argv)

    try:
        uri, database, uri_source = connection_settings()
        query = json.loads(args.query) if args.query else None
    except (ExportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    client = connect(uri, args.database or database)
    try:
        manifest = export_countries(client, args.countries, args.output, limit=args.limit,
                                    batch_size=args.batch_size, query=query,
                                    project=args.project, uri_source=uri_source,
                                    dataset=args.dataset)
    finally:
        client.client.close()
    print(f"exported {manifest['records']} records ({manifest['distinct_media']} distinct media) "
          f"-> {args.output}")
    for short in args.countries:
        counts = manifest["record_counts"][short]
        info = manifest["inputs"][short]
        state = "ok" if info.get("available", True) else f"NO SOURCE: {info.get('reason')}"
        print(f"  {short}: {counts['exported']:5} records  read {counts['read']:5}  "
              f"duplicates {counts['duplicates']:5}  invalid {counts['invalid_ids']:3}  [{state}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
