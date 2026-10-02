"""Project EP24 CSV into RDF without changing historical analytical values.

Uses only the Python standard library. Read docs/EP24_RDF.md for the contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

VERSION = "ep24-rdf-1"
NS = "https://w3id.org/laclaugpt/ep24/"
PROV = "http://www.w3.org/ns/prov#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
LOG = logging.getLogger("roihu_csv_rdf")
EXTRA_COLUMNS = ["rdf_document_uri", "rdf_row_uri", "rdf_schema_version", "rdf_status"]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def uri(base: str, project: str, kind: str, key: str) -> str:
    """Project-scoped, opaque stable identity; no names embedded in URIs."""
    return f"{base.rstrip('/')}/{quote(project, safe='')}/{kind}/{digest(key)}"


def literal(value: str) -> str:
    # N-Triples accepts \uXXXX escapes for all control characters.
    result = []
    for char in value:
        if char in ('"', "\\"):
            result.append("\\" + char)
        elif ord(char) < 32 or ord(char) == 127:
            result.append(f"\\u{ord(char):04X}")
        else:
            result.append(char)
    return '"' + "".join(result) + '"'


def triple(subject: str, predicate: str, value: str, *, resource: bool = False) -> str:
    obj = f"<{value}>" if resource else literal(value)
    return f"<{subject}> <{predicate}> {obj} .\n"


def document_key(row: dict[str, str], dataset: str, row_number: int) -> tuple[str, str]:
    # Never parse identifiers as numbers: leading zeroes and large video IDs matter.
    for column in ("_storage_id", "video_id", "document_id", "videoId", "video_filename", "source_url", "url"):
        if row.get(column, "").strip():
            return json.dumps([dataset, column, row[column]], ensure_ascii=False), column
    return json.dumps([dataset, "row", row_number]), "row-number-fallback"


def project_row(row: dict[str, str], *, base: str, project: str, dataset: str,
                row_number: int) -> tuple[str, str, str, list[str]]:
    """Lossless cells plus conservative, explicitly labelled analytical projections."""
    key, strategy = document_key(row, dataset, row_number)
    document = uri(base, project, "document", key)
    record = uri(base, project, "row", json.dumps([dataset, row_number]))
    lines = [triple(document, RDF_TYPE, NS + "Document", resource=True),
             triple(record, RDF_TYPE, NS + "CSVRow", resource=True),
             triple(record, NS + "document", document, resource=True),
             triple(record, NS + "identityStrategy", strategy),
             triple(record, NS + "rowNumber", str(row_number))]
    warnings = []
    if strategy == "row-number-fallback":
        warnings.append("No stable source ID; row identity changes if input is reordered")
    for column, value in row.items():
        cell = uri(base, project, "cell", json.dumps([dataset, row_number, column]))
        lines.extend([triple(record, NS + "cell", cell, resource=True),
                      triple(cell, RDF_TYPE, NS + "CSVCell", resource=True),
                      triple(cell, NS + "column", column),
                      triple(cell, NS + "value", value),
                      triple(cell, PROV + "wasDerivedFrom", record, resource=True)])

    # The legacy contract is one element^affect pair per line. This is a coding
    # assertion, not a claim that an actor really feels or supports something.
    for column, category in (("formula_of_populism_us", "us"),
                             ("formula_of_populism_frontier", "frontier")):
        for index, entry in enumerate(row.get(column, "").splitlines()):
            if not entry.strip():
                continue
            if entry.count("^") > 1:
                warnings.append(f"{column} line {index + 1}: malformed pair retained as raw cell")
                continue
            if "^" in entry:
                element, affect = (part.strip() for part in entry.split("^", 1))
                if not element or not affect:
                    warnings.append(f"{column} line {index + 1}: malformed pair retained as raw cell")
                    continue
            else:
                element, affect = entry.strip(), ""
            assertion = uri(base, project, "coding", json.dumps([dataset, row_number, column, index]))
            lines.extend([triple(assertion, RDF_TYPE, NS + "LaclauCoding", resource=True),
                          triple(record, NS + "coding", assertion, resource=True),
                          triple(assertion, NS + "category", category),
                          triple(assertion, NS + "element", element)])
            if affect:
                lines.append(triple(assertion, NS + "affect", affect))
            lines.extend([triple(assertion, NS + "assertionKind", "coded-origin-unspecified"),
                          triple(assertion, PROV + "wasDerivedFrom", record, resource=True)])
    return document, record, "".join(lines), warnings


def validate_header(header: list[str] | None) -> list[str]:
    if not header or any(not name for name in header) or len(set(header)) != len(header):
        raise ValueError("CSV must have nonempty, unique column names")
    if set(header) & set(EXTRA_COLUMNS):
        raise ValueError("Input already contains reserved rdf_* output columns; use the source CSV")
    return header


def export(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory to preserve edits")
    if not source.is_file():
        raise ValueError(f"Input CSV does not exist: {source}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    parsed = urlsplit(args.base_uri)
    if (parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.query
            or parsed.fragment or any(c.isspace() or c in '<>"{}|^`\\' for c in args.base_uri)):
        raise ValueError("--base-uri must be a plain absolute HTTP(S) URI without query/fragment")
    if not args.project.strip() or not args.dataset_id.strip():
        raise ValueError("Project and dataset IDs must not be empty")
    csv.field_size_limit(sys.maxsize)
    # Snapshot to disk, not RAM: later edits to the source cannot create a mixed export.
    # The output directory is deliberately exclusive to this run.
    LOG.info("stage=%s input=%s output=%s checkpoint=%s model=none",
             VERSION, source, output, args.checkpoint)
    if args.dry_run:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            header = validate_header(csv.DictReader(stream).fieldnames)
        LOG.info("dry-run validated %s columns; no outputs created", len(header))
        return {"status": "dry-run", "columns": header}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source.csv"
    file_hash = hashlib.sha256()
    with source.open("rb") as incoming, snapshot.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            file_hash.update(chunk)
            outgoing.write(chunk)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = json.dumps([VERSION, code_hash, args.base_uri, args.project, args.dataset_id])
    run_id = str(uuid.uuid4())
    run = uri(args.base_uri, args.project, "run", run_id)
    source_uri = uri(args.base_uri, args.project, "source-file", file_hash.hexdigest())
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    count = warnings_count = hits = 0
    try:
        with sqlite3.connect(args.checkpoint, timeout=60) as db:
            db.execute("CREATE TABLE IF NOT EXISTS rdf_rows_v1 "
                       "(fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            with (snapshot.open(encoding="utf-8-sig", newline="") as incoming,
                  (output / "graph.nt.partial").open("w", encoding="utf-8") as graph,
                  (output / "final.csv.partial").open("w", encoding="utf-8", newline="") as final,
                  (output / "warnings.jsonl").open("w", encoding="utf-8") as diagnostics):
                reader = csv.DictReader(incoming)
                columns = validate_header(reader.fieldnames)
                writer = csv.DictWriter(final, fieldnames=columns + EXTRA_COLUMNS)
                writer.writeheader()
                graph.write(triple(run, RDF_TYPE, PROV + "Activity", resource=True))
                graph.write(triple(run, NS + "stageVersion", VERSION))
                graph.write(triple(run, NS + "codeSHA256", code_hash))
                graph.write(triple(run, PROV + "used", source_uri, resource=True))
                graph.write(triple(source_uri, NS + "sha256", file_hash.hexdigest()))
                for number, row in enumerate(reader, start=1):
                    if args.limit is not None and number > args.limit:
                        break
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"CSV row {number}: inconsistent number of fields")
                    fingerprint = digest(config + json.dumps([number, row], ensure_ascii=False))
                    cached = db.execute("SELECT payload FROM rdf_rows_v1 WHERE fingerprint=?",
                                        (fingerprint,)).fetchone()
                    if cached:
                        document, record, triples, warnings = json.loads(cached[0])
                        hits += 1
                    else:
                        document, record, triples, warnings = project_row(
                            row, base=args.base_uri, project=args.project,
                            dataset=args.dataset_id, row_number=number)
                        db.execute("INSERT OR REPLACE INTO rdf_rows_v1 VALUES (?, ?)",
                                   (fingerprint, json.dumps([document, record, triples, warnings])))
                        db.commit()
                    graph.write(triples)
                    graph.write(triple(record, PROV + "wasDerivedFrom", source_uri, resource=True))
                    graph.write(triple(record, PROV + "wasGeneratedBy", run, resource=True))
                    writer.writerow({**row, "rdf_document_uri": document, "rdf_row_uri": record,
                                     "rdf_schema_version": VERSION,
                                     "rdf_status": "warning" if warnings else "ok"})
                    for warning in warnings:
                        diagnostics.write(json.dumps({"row": number, "warning": warning}) + "\n")
                        LOG.warning("row=%s %s", number, warning)
                    warnings_count += len(warnings)
                    count += 1
                    LOG.info("row=%s id=%s cache=%s status=%s", number, document,
                             "hit" if cached else "miss", "warning" if warnings else "ok")
            for name in ("graph.nt", "final.csv"):
                (output / (name + ".partial")).rename(output / name)
            # N-Triples is a Turtle subset. Keep the historical .nt artifact and
            # expose the same deterministic graph as .ttl for RDF tooling.
            shutil.copyfile(output / "graph.nt", output / "graph.ttl")
        manifest = {"status": "complete-with-warnings" if warnings_count else "complete",
                    "stage": VERSION, "run_id": run_id, "completed_at": datetime.now(UTC).isoformat(),
                    "source": str(source), "source_sha256": file_hash.hexdigest(),
                    "code_sha256": code_hash, "project": args.project, "dataset_id": args.dataset_id,
                    "base_uri": args.base_uri, "rows": count, "warnings": warnings_count,
                    "cache_hits": hits, "limit": args.limit, "elapsed_seconds": time.monotonic() - started}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        LOG.info("complete rows=%s warnings=%s cache_hits=%s elapsed=%.2fs outputs=%s",
                 count, warnings_count, hits, manifest["elapsed_seconds"], output)
        return manifest
    except Exception:
        LOG.exception("Export incomplete at %s; checkpoint retained. Retry with a NEW output directory", output)
        raise


def main(argv: list[str] | None = None) -> int:
    root = Path(os.environ.get("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "ep24_fi.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "rdf" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--checkpoint", type=Path, default=root / "database" / "rdf.sqlite3")
    parser.add_argument("--project", default="ep24")
    parser.add_argument("--dataset-id", default=None, help="Stable corpus name; defaults to input file stem")
    parser.add_argument("--base-uri", default="https://data.example/laclaugpt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_id = args.dataset_id or args.input.stem
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        export(args)
        orchestrated_output = os.getenv("LACLAUGPT_OUTPUT_CSV")
        if orchestrated_output and not args.dry_run:
            target = Path(orchestrated_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.output_dir / "final.csv", target)
    except (OSError, ValueError, csv.Error, sqlite3.Error) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
