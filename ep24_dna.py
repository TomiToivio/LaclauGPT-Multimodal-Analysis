"""Leifeld DNA / rDNA compatibility layer for the EP24 Roihu pipeline (issue #182).

This module is the bridge between LaclauGPT's structured discourse analysis and
the Leifeld Lab Discourse Network Analyzer ecosystem:

    Leifeld, Philip (2017). Discourse Network Analysis: Policy Debates as Dynamic
    Networks. In: Victor, Lubell & Montgomery (eds.), The Oxford Handbook of
    Political Networks, ch. 25. Oxford University Press.
    Preprint: https://eprints.gla.ac.uk/121525/

The unit of analysis is the **statement**: one actor making one claim about one
concept at one time, with an explicit qualifier where the source supports one.

Design rules
------------
1. An event-list CSV is an interchange format for rDNA, not a new network format.
   rDNA's ``dna_network(networkType="eventlist")`` *assembles* an event list from
   statements in DNA's database; feeding our statements through the documented
   batch API (``dna_addDocuments`` + ``dna_addStatement``) is therefore the
   supported path. We generate that importer rather than writing ``.dna`` files
   (``sample.dna`` is a password-protected SQLite DB and reverse-engineering it is
   explicitly out of scope).
2. Nothing here fabricates data. An ambiguous qualifier is never collapsed into
   support/oppose: the statement stays in the richer JSON with ``AGREEMENT_UNCERTAIN``
   and is excluded from the strict rDNA event list.
3. The network helpers reproduce Leifeld's published semantics deterministically.
   They are a reproducibility/verification layer, not a replacement for rDNA.

All helpers are pure functions over the event list so they are testable without
R, a database, or a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The statement type name expected by rDNA's ``dna_network`` / ``dna_addStatement``.
DNA_STATEMENT_TYPE = "DNA Statement"
# The default variable names rDNA uses for a DNA Statement type.
DNA_VAR_ACTOR = "organization"
DNA_VAR_CONCEPT = "concept"
DNA_VAR_QUALIFIER = "agreement"

# --- agreement qualifier ----------------------------------------------------
#: rDNA convention for a binary qualifier.
AGREEMENT_POSITIVE = 1
AGREEMENT_NEGATIVE = 0
#: Internal-only status for statements whose stance is not explicit enough to
#: code. These are NEVER exported as a binary qualifier.
AGREEMENT_UNCERTAIN = "uncertain"

#: Free-text stance -> canonical binary qualifier. Only unambiguous markers map.
_POSITIVE_MARKERS = frozenset({
    "support", "supports", "supportive", "endorse", "endorses", "endorsement",
    "back", "backs", "favour", "favours", "favor", "favors", "promote", "promotes",
    "defend", "defends", "praise", "praises", "approve", "approves", "agree", "agrees",
})
_NEGATIVE_MARKERS = frozenset({
    "oppose", "opposes", "opposition", "against", "reject", "rejects", "denounce",
    "denounces", "criticise", "criticises", "criticize", "criticizes", "condemn",
    "condemns", "attack", "attacks", "blame", "blames", "disapprove", "disapproves",
    "disagree", "disagrees", "resist", "resists",
})
#: Markers that explicitly mean "we do not know" rather than a stance.
_UNCERTAIN_MARKERS = frozenset({
    "neutral", "mixed", "unknown", "unclear", "ambiguous", "uncertain",
    "uncommitted", "no position", "position unclear", "",
})

#: rDNA ``qualifierAggregation`` values.
QUALIFIER_AGGREGATIONS = ("ignore", "congruence", "conflict", "subtract", "combine")
#: rDNA ``normalization`` values. ``activity``/``prominence`` are two-mode only.
NORMALIZATIONS_ONE_MODE = ("no", "average", "jaccard", "cosine")
NORMALIZATIONS_TWO_MODE = ("no", "activity", "prominence")
#: rDNA ``duplicates`` values.
DUPLICATE_POLICIES = ("include", "document", "week", "month", "year", "acrossrange")

#: Column order of the rDNA event-list interchange file.
EVENT_LIST_COLUMNS: tuple[str, ...] = (
    "statement_id",
    "date_time",
    "organization",
    "concept",
    "agreement",
    "statement_type",
    "document_id",
    "document_title",
    "document_source",
    "document_type",
    "actor_id",
    "concept_id",
    "proposition",
    "evidence_quote",
    "confidence",
    "provenance",
)

#: Columns Step 7 appends to the cumulative dataframe.
DNA_COLUMNS: tuple[str, ...] = (
    "dna_analysis_markdown",
    "dna_statements_json",
    "dna_prompt_version",
    "dna_model_metadata_json",
    "dna_generated_at",
    "dna_context_sha256",
    "dna_context_truncated",
    "dna_statement_count",
    "dna_binary_count",
    "dna_uncertain_count",
    "dna_actor_unresolved_count",
    "dna_concept_novel_count",
    "dna_codebook_fingerprint",
    "dna_memory_context_json",
    "dna_rag_context_json",
    "dna_raw_response",
    "dna_status",
    "dna_error",
    "dna_runtime_seconds",
    "dna_persistence_status",
)


def canonical_agreement(value: Any) -> int | None:
    """Map a free-text stance onto rDNA's binary qualifier.

    Returns ``1`` for explicit support, ``0`` for explicit opposition, and
    ``None`` when the stance is not explicit enough to code. ``None`` is a real
    answer: forcing an ambiguous statement into positive/negative is the exact
    distortion DNA's qualifier exists to prevent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return AGREEMENT_POSITIVE if value else AGREEMENT_NEGATIVE
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (AGREEMENT_POSITIVE, AGREEMENT_NEGATIVE):
            return value
        return None
    text = str(value).strip().casefold()
    if not text or text in _UNCERTAIN_MARKERS:
        return None
    tokens = set(re.findall(r"[a-z]+", text))
    positive = bool(tokens & _POSITIVE_MARKERS)
    negative = bool(tokens & _NEGATIVE_MARKERS)
    if positive == negative:  # neither, or both (e.g. "support and oppose")
        return None
    return AGREEMENT_POSITIVE if positive else AGREEMENT_NEGATIVE


def agreement_label(value: Any) -> str:
    """Human-readable form of a qualifier, for Markdown and logs."""
    canonical = canonical_agreement(value)
    if canonical == AGREEMENT_POSITIVE:
        return "support"
    if canonical == AGREEMENT_NEGATIVE:
        return "oppose"
    return AGREEMENT_UNCERTAIN


def parse_date_time(value: Any) -> str:
    """Normalise a statement timestamp, preserving precision actually present.

    Leifeld's method is explicitly longitudinal, so the timestamp must survive.
    A full ISO-8601 value is preserved to the microsecond; weaker values are
    preserved at the precision they carry rather than being invented.
    """
    if value in (None, ""):
        return ""
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y-%m", "%Y"):
            try:
                parsed = datetime.strptime(text, pattern)
            except ValueError:
                continue
            break
        else:
            return ""  # Unparseable: do not invent a timestamp.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def duplicate_bucket(timestamp: str, policy: str) -> str:
    """Return the time bucket rDNA's ``duplicates`` setting would collapse into.

    Mirrors the semantics documented on ``dna_network``: ``document`` keeps one
    statement per document, ``week``/``month``/``year`` per calendar period, and
    ``acrossrange`` keeps exactly one across the whole range.
    """
    if policy not in DUPLICATE_POLICIES:
        raise ValueError(f"unknown duplicate policy {policy!r}; choose one of {DUPLICATE_POLICIES}")
    if policy == "include":
        return ""
    if policy == "acrossrange":
        return "*"
    stamp = parse_date_time(timestamp)
    if not stamp:
        return ""
    date = datetime.fromisoformat(stamp)
    if policy == "week":
        iso = date.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if policy == "month":
        return f"{date.year}-{date.month:02d}"
    if policy == "year":
        return f"{date.year}"
    return ""  # "document" is bucketed by the caller with the document id.


def deduplicate(statements: Sequence[Mapping[str, Any]], *, policy: str = "include") -> list[dict]:
    """Apply an rDNA duplicate policy deterministically.

    Statement identity is the actor/concept/qualifier tuple, so ``include``
    keeps everything, ``document`` keeps one per (document, tuple), and the
    calendar policies keep one per (period, tuple).
    """
    if policy not in DUPLICATE_POLICIES:
        raise ValueError(f"unknown duplicate policy {policy!r}; choose one of {DUPLICATE_POLICIES}")
    if policy == "include":
        return [dict(item) for item in statements]
    seen: set[tuple[str, ...]] = set()
    kept: list[dict] = []
    for item in statements:
        tuple_key = (
            str(item.get("organization", "")).casefold(),
            str(item.get("concept", "")).casefold(),
            str(item.get("agreement", "")),
        )
        prefix = str(item.get("document_id", "")) if policy == "document" else ""
        bucket = duplicate_bucket(str(item.get("date_time", "")), policy)
        key = (prefix, bucket, *tuple_key)
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(item))
    return kept


def _matrix_axis(statements: Iterable[Mapping[str, Any]], variable: str) -> list[str]:
    axis = {str(item.get(variable, "")) for item in statements if str(item.get(variable, ""))}
    return sorted(axis)


def two_mode_network(
    statements: Sequence[Mapping[str, Any]],
    *,
    variable1: str = DNA_VAR_ACTOR,
    variable2: str = DNA_VAR_CONCEPT,
    qualifier: str | None = DNA_VAR_QUALIFIER,
    qualifier_aggregation: str = "ignore",
    normalization: str = "no",
) -> dict[str, Any]:
    """Build an actor x concept affiliation matrix with rDNA semantics.

    ``qualifier_aggregation`` follows ``dna_network`` for two-mode networks:
    ``ignore`` counts every statement, ``combine`` multiplexes positive/negative
    into ``1``/``2``/``3``, and ``subtract`` subtracts negative from positive.
    """
    if qualifier_aggregation not in QUALIFIER_AGGREGATIONS:
        raise ValueError(f"unknown qualifier aggregation {qualifier_aggregation!r}")
    if normalization not in NORMALIZATIONS_TWO_MODE:
        raise ValueError(f"unknown two-mode normalization {normalization!r}")
    rows = _matrix_axis(statements, variable1)
    columns = _matrix_axis(statements, variable2)
    row_index = {name: i for i, name in enumerate(rows)}
    col_index = {name: i for i, name in enumerate(columns)}
    counts = [[0 for _ in columns] for _ in rows]
    positives = [[0 for _ in columns] for _ in rows]
    negatives = [[0 for _ in columns] for _ in rows]
    for item in statements:
        row_name, col_name = str(item.get(variable1, "")), str(item.get(variable2, ""))
        if not row_name or not col_name:
            continue
        i, j = row_index[row_name], col_index[col_name]
        counts[i][j] += 1
        canonical = canonical_agreement(item.get(qualifier)) if qualifier else None
        if canonical == AGREEMENT_POSITIVE:
            positives[i][j] += 1
        elif canonical == AGREEMENT_NEGATIVE:
            negatives[i][j] += 1

    if qualifier_aggregation == "subtract" and qualifier:
        values = [[positives[i][j] - negatives[i][j] for j in range(len(columns))]
                  for i in range(len(rows))]
    elif qualifier_aggregation == "combine" and qualifier:
        values = []
        for i in range(len(rows)):
            row: list[int] = []
            for j in range(len(columns)):
                if positives[i][j] and negatives[i][j]:
                    row.append(3)  # mixed
                elif positives[i][j]:
                    row.append(1)  # positive only
                elif negatives[i][j]:
                    row.append(2)  # negative only
                else:
                    row.append(0)  # uncoded
            values.append(row)
    else:
        values = [row[:] for row in counts]

    if normalization in {"activity", "prominence"} and rows and columns:
        row_totals = [sum(row) for row in values]
        col_totals = [sum(values[i][j] for i in range(len(rows))) for j in range(len(columns))]
        grand = sum(row_totals)
        if grand:
            divisors = row_totals if normalization == "activity" else col_totals
            values = [
                [
                    (values[i][j] * grand) / divisors[i]
                    if normalization == "activity" and divisors[i]
                    else (values[i][j] * grand) / divisors[j]
                    if normalization == "prominence" and divisors[j]
                    else 0.0
                    for j in range(len(columns))
                ]
                for i in range(len(rows))
            ]
    return {
        "network_type": "twomode",
        "variable1": variable1,
        "variable2": variable2,
        "qualifier_aggregation": qualifier_aggregation if qualifier_aggregation != "combine" else "combine",
        "normalization": normalization,
        "rows": rows,
        "columns": columns,
        "values": values,
    }


def _dyad_metric(
    profiles: Mapping[str, Mapping[str, int]],
    *,
    mode: str,
) -> list[tuple[str, str, float]]:
    """Shared driver for congruence (same stance) and conflict (opposite stance).

    ``profiles`` maps a node to ``{key: qualifier}``. Two nodes are compared over
    the keys they *share*: congruence counts a shared key where the qualifiers
    agree, conflict counts a shared key where they differ. Keys are therefore the
    plain concept (or actor) identity, never the signed concept, or the two sides
    of a disagreement would never meet.
    """
    names = sorted(profiles)
    edges: list[tuple[str, str, float]] = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            weight = 0.0
            for key in set(profiles[left]) & set(profiles[right]):
                same = profiles[left][key] == profiles[right][key]
                if (mode == "congruence" and same) or (mode == "conflict" and not same):
                    weight += 1
            if weight:
                edges.append((left, right, weight))
    return edges


def actor_stances(statements: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Map actor -> concept -> canonical qualifier, ignoring uncoded statements.

    The key is the plain concept so that two actors taking opposite stances on the
    same concept are comparable; the qualifier is the value that distinguishes
    them.
    """
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for item in statements:
        actor = str(item.get("organization", ""))
        concept = str(item.get("concept", ""))
        canonical = canonical_agreement(item.get("agreement"))
        if not actor or not concept or canonical is None:
            continue
        result[actor][concept] = canonical
    return {actor: dict(concepts) for actor, concepts in result.items()}


def actor_congruence_network(statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, float]]:
    """Leifeld actor congruence: a tie when actors code a shared concept identically.

    Concept + qualifier are treated as a unit of comparison, so two actors who
    both *support* the same concept are congruent, while an actor supporting and
    one opposing it are not (they form a conflict tie instead).
    """
    return _dyad_metric(actor_stances(statements), mode="congruence")


def actor_conflict_network(statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, float]]:
    """Leifeld actor conflict: a tie when actors take opposite stances on a concept."""
    return _dyad_metric(actor_stances(statements), mode="conflict")


def apply_normalization(
    edges: Sequence[tuple[str, str, float]],
    *,
    statements: Sequence[Mapping[str, Any]],
    normalization: str,
) -> list[tuple[str, str, float]]:
    """Normalize congruence/conflict edge weights the way rDNA does.

    ``average`` divides by average activity, ``jaccard`` by the union of concepts
    each actor coded, ``cosine`` by the geometric mean of the two actors' concept
    counts. Raw activity is never a measure of ideological similarity, which is
    exactly why this step exists.
    """
    if normalization not in NORMALIZATIONS_ONE_MODE:
        raise ValueError(f"unknown one-mode normalization {normalization!r}")
    if normalization == "no":
        return list(edges)
    counts: dict[str, int] = defaultdict(int)
    concepts: dict[str, set[str]] = defaultdict(set)
    for item in statements:
        actor, concept = str(item.get("organization", "")), str(item.get("concept", ""))
        canonical = canonical_agreement(item.get("agreement"))
        if not actor or not concept or canonical is None:
            continue
        counts[actor] += 1
        concepts[actor].add(f"{concept}|{canonical}")
    out: list[tuple[str, str, float]] = []
    for left, right, weight in edges:
        if normalization == "average":
            divisor = (counts[left] + counts[right]) / 2
        elif normalization == "jaccard":
            union = concepts[left] | concepts[right]
            divisor = len(union)
        else:  # cosine
            product = counts[left] * counts[right]
            divisor = product ** 0.5
        out.append((left, right, weight / divisor if divisor else 0.0))
    return out


def signed_concepts(statements: Sequence[Mapping[str, Any]]) -> list[str]:
    """Concept + qualifier tuples as distinct signed concepts (Leifeld)."""
    signed = {
        f"{item.get('concept', '')}|{canonical_agreement(item.get('agreement'))}"
        for item in statements
        if str(item.get("concept", ""))
        and canonical_agreement(item.get("agreement")) is not None
    }
    return sorted(signed)


def concept_congruence_pairs(statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, float]]:
    """Concept x concept congruence via shared actor stance."""
    by_concept: dict[str, dict[str, int]] = defaultdict(dict)
    for item in statements:
        actor = str(item.get("organization", ""))
        concept = str(item.get("concept", ""))
        canonical = canonical_agreement(item.get("agreement"))
        if not actor or not concept or canonical is None:
            continue
        by_concept[f"{concept}|{canonical}"][actor] = canonical
    return _dyad_metric(by_concept, mode="congruence")


def statement_id(
    *,
    source_record_id: str,
    actor: str,
    concept: str,
    agreement: int | None,
    date_time: str,
    proposition: str = "",
) -> str:
    """Deterministic statement identity, stable across runs and reorderings.

    Stable IDs are what make rDNA's duplicate policies meaningful: the same
    actor saying the same thing about the same concept in the same document must
    hash to the same statement.
    """
    payload = json.dumps(
        [source_record_id, actor, concept, agreement, date_time, proposition],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_event_list(
    statements: Sequence[Mapping[str, Any]],
    *,
    document_id: str,
    document_title: str = "",
    document_source: str = "",
    document_type: str = "EP24 social media video",
    duplicate_policy: str = "include",
) -> list[dict]:
    """Project statements into the rDNA event-list interchange shape.

    Only statements with a canonical binary qualifier are exported: rDNA's
    qualifier is binary, and dropping the ambiguous cases from the strict export
    (while keeping them in ``dna_statements_json``) is the documented policy.
    """
    rows: list[dict] = []
    for item in statements:
        canonical = canonical_agreement(item.get("agreement"))
        if canonical is None:
            continue
        rows.append({
            "statement_id": str(item.get("statement_id", "")),
            "date_time": str(item.get("date_time", "")),
            "organization": str(item.get("organization", "")),
            "concept": str(item.get("concept", "")),
            "agreement": canonical,
            "statement_type": DNA_STATEMENT_TYPE,
            "document_id": document_id,
            "document_title": document_title,
            "document_source": document_source,
            "document_type": document_type,
            "actor_id": str(item.get("actor_id", "")),
            "concept_id": str(item.get("concept_id", "")),
            "proposition": str(item.get("proposition", "")),
            "evidence_quote": str(item.get("evidence_quote", "")),
            "confidence": item.get("confidence", ""),
            "provenance": str(item.get("provenance", "")),
        })
    return deduplicate(rows, policy=duplicate_policy)


def write_event_list_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the event-list interchange file deterministically (sorted rows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("date_time", "")),
            str(row.get("organization", "")),
            str(row.get("concept", "")),
            str(row.get("statement_id", "")),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EVENT_LIST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return path


def write_network_csv(network: Mapping[str, Any], path: str | Path) -> Path:
    """Write a two-mode matrix in the long form DNA/UCINET tooling can read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = network.get("rows", [])
    columns = network.get("columns", [])
    values = network.get("values", [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([str(network.get("variable1", "actor")), str(network.get("variable2", "concept")), "weight"])
        for i, row_name in enumerate(rows):
            for j, col_name in enumerate(columns):
                weight = values[i][j] if i < len(values) and j < len(values[i]) else 0
                if weight:
                    writer.writerow([row_name, col_name, weight])
    return path


def write_edge_csv(edges: Sequence[tuple[str, str, float]], path: str | Path, *, label: str = "weight") -> Path:
    """Write a one-mode edge list deterministically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "target", label])
        for left, right, weight in sorted(edges):
            writer.writerow([left, right, weight])
    return path


def write_graphml(edges: Sequence[tuple[str, str, float]], path: str | Path, *, label: str = "weight") -> Path:
    """Emit GraphML so DNA's own downstream export target (visone etc.) works."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = sorted({node for left, right, _ in edges for node in (left, right)})
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        f'  <key id="{label}" for="edge" attr.name="{label}" attr.type="double"/>',
        '  <graph edgedefault="undirected">',
    ]
    for node in nodes:
        lines.append(f'    <node id="{_xml_escape(node)}"/>')
    for index, (left, right, weight) in enumerate(sorted(edges), start=1):
        lines.append(
            f'    <edge id="e{index}" source="{_xml_escape(left)}" '
            f'target="{_xml_escape(right)}"><data key="{label}">{weight}</data></edge>'
        )
    lines.extend(['  </graph>', '</graphml>'])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_rdna_import_script(
    event_list_path: str | Path,
    path: str | Path,
    *,
    database_url: str = "sqlite:///ep24.dna",
    coder_password: str = "",
) -> Path:
    """Generate an R script that imports the event list through the official API.

    This is the documented compatibility path: rDNA's batch API
    (``dna_addDocuments`` + ``dna_addStatement``) rather than fabricating a
    ``.dna`` SQLite file. The generated script is data, not code we run here, so
    the pipeline stays runnable without R.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(event_list_path).name
    script = f'''# Generated by LaclauGPT Step 7 (issue #182). Imports EP24 DNA statements into DNA.
#
# Route: the official rDNA batch API. We deliberately do not write .dna files:
# sample.dna is a password-protected SQLite database and reverse-engineering it
# is unsupported. See https://github.com/leifeld-lab/dna (rDNA/rDNA/R/rDNA.R).
#
#   Rscript {Path(path).name}
#
# Methodology: Leifeld, P. (2017). Discourse Network Analysis: Policy Debates as
# Dynamic Networks. In: The Oxford Handbook of Political Networks, ch. 25.

library(rDNA)

events <- read.csv("{source}", stringsAsFactors = FALSE, encoding = "UTF-8")

dna_init()
dna_openDatabase("{database_url}", coderPassword = "{coder_password}")

# One DNA document per EP24 source record.
documents <- unique(events[, c("document_id", "document_title", "document_source", "document_type", "date_time")])
doc_ids <- integer(nrow(documents))
for (i in seq_len(nrow(documents))) {{
  doc_ids[i] <- dna_addDocuments(
    coder_id = 1,
    title = documents$document_title[i],
    text = documents$document_source[i],
    source = documents$document_type[i],
    date_time = as.POSIXct(documents$date_time[i], tz = "UTC")
  )
}}
lookup <- setNames(doc_ids, documents$document_id)

# One DNA statement per event: organization + concept + agreement (+ document).
for (i in seq_len(nrow(events))) {{
  dna_addStatement(
    documentID = lookup[[events$document_id[i]]],
    statementType = "{DNA_STATEMENT_TYPE}",
    organization = events$organization[i],
    concept = events$concept[i],
    agreement = as.integer(events$agreement[i])
  )
}}

dna_printDetails()
dna_closeDatabase()

# After import, build networks with rDNA's own semantics, e.g.:
#   dna_network(networkType = "twomode", statementType = "{DNA_STATEMENT_TYPE}",
#               variable1 = "{DNA_VAR_ACTOR}", variable2 = "{DNA_VAR_CONCEPT}",
#               qualifier = "{DNA_VAR_QUALIFIER}", qualifierAggregation = "subtract")
#   dna_network(networkType = "onemode", variable1 = "{DNA_VAR_ACTOR}",
#               variable2 = "{DNA_VAR_CONCEPT}", qualifier = "{DNA_VAR_QUALIFIER}",
#               qualifierAggregation = "congruence", normalization = "jaccard")
'''
    path.write_text(script, encoding="utf-8")
    return path


def read_event_list_csv(path: str | Path) -> list[dict]:
    """Read an event-list interchange file, coercing the qualifier back to int."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        raw = str(row.get("agreement", "")).strip()
        row["agreement"] = int(raw) if raw.lstrip("-").isdigit() else raw
        raw_confidence = str(row.get("confidence", "")).strip()
        if raw_confidence:
            try:
                row["confidence"] = float(raw_confidence)
            except ValueError:
                pass
    return rows


def summarize(statements: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts used in logs and in the ``dna_*_count`` contract columns."""
    binary = sum(1 for item in statements if canonical_agreement(item.get("agreement")) is not None)
    return {
        "statement_count": len(statements),
        "binary_count": binary,
        "uncertain_count": len(statements) - binary,
        "actor_unresolved_count": sum(
            1 for item in statements if str(item.get("actor_id", "")).strip() == ""
        ),
        "concept_novel_count": sum(
            1 for item in statements if str(item.get("concept_provenance", "")) == "novel"
        ),
    }


def iter_statements(value: Any) -> Iterator[dict]:
    """Tolerantly parse a ``dna_statements_json`` cell into statement dicts."""
    if value in (None, ""):
        return
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(parsed, list):
        return
    for item in parsed:
        if isinstance(item, dict):
            yield item


def empty_statements() -> list[dict[str, Any]]:
    """Documented empty shape, so callers never have to special-case None."""
    return []


def dataclass_dict(item: Any) -> dict[str, Any]:
    """Serialize a pydantic model or dataclass into a plain dict."""
    if hasattr(item, "model_dump"):
        return dict(item.model_dump())
    if hasattr(item, "__dataclass_fields__"):
        return asdict(item)
    return dict(item)


@dataclass(frozen=True)
class EventListSummary:
    """Small typed summary for tests and manifests."""

    statements: int = 0
    exported: int = 0
    uncertain: int = 0
    actors: tuple[str, ...] = field(default_factory=tuple)
    concepts: tuple[str, ...] = field(default_factory=tuple)
