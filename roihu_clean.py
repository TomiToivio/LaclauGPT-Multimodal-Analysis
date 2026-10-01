#!/usr/bin/env python3
"""Non-destructive cleaning of legacy EP24 per-country dataframes (issue #35).

The legacy per-country CSVs are research artifacts and publication provenance
sources. This module never rewrites them. It reads a private source CSV and
emits *derivative* artifacts next to it:

    ep24_<country>_cleaned.csv              legacy columns + additive cleaning fields
    ep24_<country>_cleaning_decisions.csv   compact row-level decision log
    ep24_<country>_cleaning_report.md       human-readable audit
    ep24_<country>_cleaning_provenance.json machine-readable provenance
    data/to_reprocess/ep24_<country>.csv    new-pipeline input, keep-schema only
    data/ep24_videos_to_recut.csv           cross-country recut/split worklist

Design rules enforced here (issue #35, all comments):

* nothing is destroyed — the cleaned derivative keeps every source row and every
  legacy column unless an explicit, documented rule removes a column;
* blank is not false — researcher tri-state is parsed strictly, and an
  unrecognised value raises instead of being coerced;
* no silent row loss — rows are marked ``include_in_reprocess`` and never dropped
  from the derivative; explicit exclusions are recorded with a reason;
* legacy model output is never recycled as if it were human ground truth.

The module is generic across countries: country differences belong in codebooks,
not in dataframe structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --- canonical reuse: never re-declare the EP24 media skip rule -------------
try:  # pragma: no cover - import is the point
    from ep24_video import VIDEO_INITIAL_SKIP_SECONDS
except Exception:  # pragma: no cover - keeps the cleaner importable standalone
    VIDEO_INITIAL_SKIP_SECONDS = 1.0

# --- authoritative keep-schema (issue #35 comments [5] and [6]) -------------
# One common schema for every EP24 country; order recorded from
# research_keep/ep24_finland_keep_columns.xlsx.
KEEP_SCHEMA_ORDER: tuple[str, ...] = (
    "country",
    "author_username",
    "account_type",
    "source_type",
    "source_recording",
    "video_id",
    "sequence_number",
    "political_preference",
    "allas_filename",
    "new_entity",
    "new_theme",
    "video_duration",
    "researcher_new_persons",
    "researcher_new_themes",
    "researcher_note",
)

# Comment [5] requires video_id first. Flip this single constant to follow the
# workbook order verbatim instead.
VIDEO_ID_FIRST = True

# --- obsolete legacy columns (comment [2]) ---------------------------------
# Resolved against the real header; spreadsheet letters are never trusted as
# positional indices.
OBSOLETE_LETTERS: tuple[str, ...] = ("Y", "Z", "AA", "AB", "AE")
OBSOLETE_LETTER_RANGE: tuple[str, str] = ("AH", "AT")
LDA_PREFIX = "lda_"

# --- legacy model-analysis outputs: diagnostic only, never recycled (comment [4]) ---
LEGACY_MODEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "summary_analysis",
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "formula_of_populism_us_elements",
    "formula_of_populism_frontier_elements",
    "formula_of_populism_us_affects",
    "formula_of_populism_frontier_affects",
    "entities",
    "topics",
    "positive",
    "neutral",
    "negative",
    "manifestoberta_predicted_class",
    "manifestoberta_probabilities",
    "spacy_entities",
    "us",
    "them",
)

# `researcher_rerun_*` no longer controls anything: every selected video is
# rerun anyway (comment [2] §3). Preserved in provenance, never a decision input.
RERUN_COLUMNS: tuple[str, ...] = (
    "researcher_rerun_llm",
    "researcher_rerun_ocr",
    "researcher_rerun_whisper",
)

DELETE_COLUMN = "researcher_delete_video"
DUBIOUS_COLUMN = "researcher_dubious_video"
CUT_COLUMNS: tuple[str, ...] = ("researcher_cut_video",)
SPLIT_COLUMNS: tuple[str, ...] = ("researcher_split_video",)

# --- row-level decision states (issue #35) ---------------------------------
DECISION_KEEP = "KEEP"
DECISION_KEEP_WITH_CORRECTION = "KEEP_WITH_CORRECTION"
DECISION_EXCLUDE = "EXCLUDE_FROM_NEW_ANALYSIS"
DECISION_DUBIOUS = "DUBIOUS_REVIEW"
DECISION_SPLIT = "SPLIT_REQUIRED"
DECISION_CUT = "CUT_REQUIRED"
DECISION_WRONG_LANGUAGE = "WRONG_LANGUAGE"
DECISION_RERUN = "RERUN_REQUIRED"
DECISION_DUPLICATE = "DUPLICATE_REVIEW"
DECISION_UNKNOWN = "UNKNOWN"

# --- additive cleaning fields (never overwrite a legacy value) -------------
CLEANING_FIELDS: tuple[str, ...] = (
    "cleaning_status",
    "cleaning_reason",
    "cleaning_rule_id",
    "cleaning_sources",
    "cleaning_original_value",
    "cleaning_corrected_value",
    "cleaning_decision_origin",
    "cleaning_ambiguous",
    "cleaning_notes_md",
    "include_in_reprocess",
    "needs_human_review",
    "seed_entities",
    "seed_themes",
)

# --- pseudo-country rejection (comment [1]) --------------------------------
# Country must come from country/profile metadata, never from a storage path
# component. Directory names such as `MobileBackup` leaked in as a "country".
PSEUDO_COUNTRY_PATTERNS: tuple[str, ...] = (
    r"^mobilebackup$",
    r"^mobile[_\-\s]?backup$",
    r"^backup$",
    r"^scratch$",
    r"^mobile$",
)
KNOWN_COUNTRY_CODES: tuple[str, ...] = (
    "FI",
    "SE",
    "PL",
    "PT",
    "DE",
    "ES",
    "HU",
    "HR",
    "FR",
    "BG",
)
KNOWN_COUNTRY_NAMES: tuple[str, ...] = (
    "finland",
    "sweden",
    "poland",
    "portugal",
    "germany",
    "spain",
    "hungary",
    "croatia",
    "france",
    "bulgaria",
)
TRIVIAL_NOTES: tuple[str, ...] = ("test", "testing", "test note", "just a test note!")

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


class CleaningError(RuntimeError):
    """Raised when cleaning would destroy or misinterpret research data."""


def spreadsheet_letters(index: int) -> str:
    """0-based column index -> spreadsheet letters (0 -> A, 26 -> AA)."""
    letters = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _letter_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - 64)
    return value - 1


@dataclass(frozen=True)
class ObsoleteColumns:
    """Obsolete legacy columns resolved against the real header."""

    letters: dict[str, str]
    lda: list[str]

    @property
    def resolved(self) -> list[str]:
        return list(self.letters.values()) + list(self.lda)


def resolve_obsolete_columns(header: Sequence[str]) -> ObsoleteColumns:
    """Map spreadsheet letters onto *actual* field names from the real header."""
    letters: dict[str, str] = {}
    index_of = {name: position for position, name in enumerate(header)}
    for letter in OBSOLETE_LETTERS:
        position = _letter_index(letter)
        if position >= len(header):
            raise CleaningError(f"obsolete column {letter} is past the header width")
        letters[letter] = header[position]
    start, end = (_letter_index(item) for item in OBSOLETE_LETTER_RANGE)
    if end >= len(header):
        raise CleaningError("obsolete column range AH-AT is past the header width")
    lda: list[str] = []
    for position in range(start, end + 1):
        name = header[position]
        if not name.startswith(LDA_PREFIX):
            raise CleaningError(
                f"expected an {LDA_PREFIX}* field at column "
                f"{spreadsheet_letters(position)}, found {name!r}"
            )
        lda.append(name)
    resolved = list(letters.values()) + lda
    if len(set(resolved)) != len(resolved):
        raise CleaningError("obsolete column resolution produced duplicates")
    for name in resolved:
        if name not in index_of:  # pragma: no cover - defensive
            raise CleaningError(f"resolved obsolete field {name!r} is not in the header")
    return ObsoleteColumns(letters=letters, lda=lda)


def parse_tri_state(value: Any) -> bool | None:
    """Strict researcher tri-state.

    ``None`` means blank / not reviewed / not stated. It is never a synonym for
    ``False`` (issue #35: "never treat blank as false"). An unrecognised value
    raises rather than being silently coerced.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise CleaningError(f"unrecognised researcher tri-state value: {value!r}")


def normalise_political_preference(value: Any) -> tuple[str, str, str]:
    """Return (raw, canonical, ambiguity). Unknown/blank stays distinct."""
    raw = "" if value is None else str(value).strip()
    table = {
        "centre right": "Centre right",
        "center right": "Centre right",
        "red-green": "Red-green",
        "red green": "Red-green",
        "far right": "Far right",
    }
    canonical = table.get(raw.casefold(), "")
    if canonical:
        return raw, canonical, ""
    if raw:
        return raw, "", "unrecognised_political_preference"
    return raw, "", ""


def is_pseudo_country(value: Any) -> bool:
    """True when a 'country' is really a storage directory name."""
    text = str(value or "").strip().casefold()
    if not text:
        return False
    if any(re.fullmatch(pattern, text) for pattern in PSEUDO_COUNTRY_PATTERNS):
        return True
    return False


def is_known_country(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.upper() in KNOWN_COUNTRY_CODES or text.casefold() in KNOWN_COUNTRY_NAMES


def split_multi_value(value: Any) -> list[str]:
    """Split a legacy multi-value cell. Never invents values."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    parts = text.split(";") if ";" in text else text.split(",")
    return [part.strip() for part in parts if part.strip()]


def merge_seed_values(*values: Any) -> list[str]:
    """Deterministic union of human-curated labels, de-duplicated case-insensitively."""
    seen: dict[str, str] = {}
    for value in values:
        for item in split_multi_value(value):
            key = item.casefold()
            if key not in seen:
                seen[key] = item
    return [seen[key] for key in sorted(seen)]


@dataclass
class RowDecision:
    """One row-level cleaning decision with its evidence and provenance."""

    video_id: str
    country: str
    profile: str
    platform: str
    decision: str
    reason: str
    sources: list[str] = field(default_factory=list)
    original_value: str = ""
    cleaned_value: str = ""
    origin: str = "derived"
    ambiguous: bool = False
    rule_id: str = ""
    include_in_reprocess: bool = True
    needs_human_review: bool = False
    notes_md: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "video_id": self.video_id,
            "country": self.country,
            "profile": self.profile,
            "platform": self.platform,
            "decision": self.decision,
            "reason": self.reason,
            "sources": ";".join(self.sources),
            "original_value": self.original_value,
            "cleaned_value": self.cleaned_value,
            "decision_origin": self.origin,
            "ambiguous": "true" if self.ambiguous else "false",
            "rule_id": self.rule_id,
            "include_in_reprocess": "true" if self.include_in_reprocess else "false",
            "needs_human_review": "true" if self.needs_human_review else "false",
            "cleaning_notes_md": self.notes_md,
        }


DECISION_FIELDS: tuple[str, ...] = tuple(
    RowDecision(video_id="", country="", profile="", platform="", decision="", reason="").as_row()
)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def compute_cleaning_result(
    row: Mapping[str, Any],
    *,
    duplicate_ids: frozenset[str] = frozenset(),
) -> RowDecision:
    """Pure decision function: one row in, one RowDecision out.

    Precedence follows issue #35: explicit human delete/dubious exclusions first,
    then structural worklist cases (cut/split), then triage of the free-text note.
    """
    video_id = _clean_cell(row.get("video_id"))
    country = _clean_cell(row.get("country"))
    profile = _clean_cell(row.get("author_username"))
    platform = _clean_cell(row.get("source_type"))

    delete = parse_tri_state(row.get(DELETE_COLUMN))
    dubious = parse_tri_state(row.get(DUBIOUS_COLUMN))
    cut = any(parse_tri_state(row.get(name)) is True for name in CUT_COLUMNS)
    split = any(parse_tri_state(row.get(name)) is True for name in SPLIT_COLUMNS)
    wrong_language = parse_tri_state(row.get("researcher_wrong_language"))
    rerun = any(parse_tri_state(row.get(name)) is True for name in RERUN_COLUMNS)
    account_type = _clean_cell(row.get("account_type"))
    note = _clean_cell(row.get("researcher_note"))

    if delete is True:
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            DECISION_EXCLUDE,
            "researcher_delete_video is explicitly TRUE",
            [DELETE_COLUMN],
            original_value=note,
            origin="human",
            rule_id="R-DELETE-VIDEO",
            include_in_reprocess=False,
            needs_human_review=False,
            notes_md="Explicit human decision; source row preserved in the legacy CSV.",
        )
    if dubious is True:
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            DECISION_DUBIOUS,
            "researcher_dubious_video is explicitly TRUE",
            [DUBIOUS_COLUMN],
            original_value=note,
            origin="human",
            rule_id="R-DUBIOUS-VIDEO",
            include_in_reprocess=False,
            needs_human_review=True,
            notes_md="Explicit human doubt; flagged for review, not silently kept.",
        )
    if cut or split:
        flags = [
            name
            for name in (*CUT_COLUMNS, *SPLIT_COLUMNS)
            if parse_tri_state(row.get(name)) is True
        ]
        decision = DECISION_SPLIT if split else DECISION_CUT
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            decision,
            "video needs recut/resplit before analysis",
            flags,
            original_value=note,
            origin="human",
            rule_id="R-RECUT-WORKLIST",
            include_in_reprocess=False,
            needs_human_review=True,
            notes_md="Moved to the shared cross-country recut worklist.",
        )
    if wrong_language is True:
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            DECISION_WRONG_LANGUAGE,
            "researcher_wrong_language is explicitly TRUE",
            ["researcher_wrong_language"],
            original_value=note,
            origin="human",
            rule_id="R-WRONG-LANGUAGE",
            include_in_reprocess=False,
            needs_human_review=True,
        )

    note_decision = _triage_note(note)
    if note_decision is not None:
        return note_decision

    if account_type and account_type.casefold() != "synthetic":
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            DECISION_EXCLUDE,
            f"account_type is {account_type!r}, not Synthetic",
            ["account_type"],
            origin="derived",
            rule_id="R-SYNTHETIC-ONLY",
            include_in_reprocess=False,
            notes_md="Organic profiles are excluded from the current analysis target.",
        )

    if video_id and video_id in duplicate_ids:
        return RowDecision(
            video_id,
            country,
            profile,
            platform,
            DECISION_DUPLICATE,
            "video_id occurs more than once in this country dataframe",
            ["video_id"],
            origin="derived",
            rule_id="R-DUPLICATE-REVIEW",
            include_in_reprocess=True,
            needs_human_review=True,
            notes_md="Duplicates are flagged for humans, never deduplicated silently.",
        )

    sources: list[str] = []
    if rerun:
        sources.extend(RERUN_COLUMNS)
    return RowDecision(
        video_id,
        country,
        profile,
        platform,
        DECISION_KEEP,
        "no explicit exclusion rule matched",
        sources,
        origin="derived",
        rule_id="R-KEEP",
        include_in_reprocess=True,
        notes_md=(
            "Legacy rerun flags observed; they no longer gate the new pipeline." if rerun else ""
        ),
    )


def _triage_note(note: str) -> RowDecision | None:
    """Conservative triage of the free-text researcher note (comment [5])."""
    if not note:
        return None
    lowered = " ".join(note.casefold().split())
    if lowered in TRIVIAL_NOTES:
        return RowDecision(
            "",
            "",
            "",
            "",
            DECISION_KEEP_WITH_CORRECTION,
            "researcher note is a trivial test string",
            ["researcher_note"],
            original_value=note,
            cleaned_value="",
            origin="derived",
            rule_id="R-NOTE-TRIVIAL",
            include_in_reprocess=True,
        )
    # Anything else is preserved verbatim as private human context. The cleaner
    # does not attempt sentiment/quality judgement on free text: an ambiguous
    # note goes to human review rather than being auto-excluded.
    return RowDecision(
        "",
        "",
        "",
        "",
        DECISION_KEEP,
        "informative researcher note preserved as private context",
        ["researcher_note"],
        original_value=note,
        cleaned_value=note,
        origin="human",
        rule_id="R-NOTE-CONTEXT",
        include_in_reprocess=True,
    )


def _apply_decision(row: dict[str, Any], decision: RowDecision) -> dict[str, Any]:
    """Attach additive cleaning fields without rewriting any legacy value."""
    enriched = dict(row)
    enriched["cleaning_status"] = decision.decision
    enriched["cleaning_reason"] = decision.reason
    enriched["cleaning_rule_id"] = decision.rule_id
    enriched["cleaning_sources"] = ";".join(decision.sources)
    enriched["cleaning_original_value"] = decision.original_value
    enriched["cleaning_corrected_value"] = decision.cleaned_value
    enriched["cleaning_decision_origin"] = decision.origin
    enriched["cleaning_ambiguous"] = "true" if decision.ambiguous else "false"
    enriched["cleaning_notes_md"] = decision.notes_md
    enriched["include_in_reprocess"] = "true" if decision.include_in_reprocess else "false"
    enriched["needs_human_review"] = "true" if decision.needs_human_review else "false"
    enriched["seed_entities"] = "; ".join(
        merge_seed_values(row.get("new_entity"), row.get("researcher_new_persons"))
    )
    enriched["seed_themes"] = "; ".join(
        merge_seed_values(row.get("new_theme"), row.get("researcher_new_themes"))
    )
    return enriched


@dataclass
class CleaningOutcome:
    country: str
    source_path: Path
    source_rows: int
    source_columns: list[str]
    cleaned_columns: list[str]
    removed_columns: dict[str, Any]
    empty_columns: list[str]
    decisions: list[RowDecision]
    rows: list[dict[str, Any]]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.decision] = counts.get(decision.decision, 0) + 1
        return counts

    def reprocess_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("include_in_reprocess") == "true"]

    def recut_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self.rows if row.get("cleaning_status") in {DECISION_CUT, DECISION_SPLIT}
        ]


def load_source(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a private source CSV without modifying it."""
    source = Path(path)
    if not source.exists():
        raise CleaningError(f"source dataframe not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not header:
        raise CleaningError(f"source dataframe has no header: {source}")
    return header, rows


def find_always_empty_columns(
    header: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Columns with no value in any row (comment [2] §2). Researcher fields are
    excluded from this audit because their sparsity is meaningful, not dead."""
    candidates = [
        name
        for name in header
        if not name.startswith("researcher_") and name not in {DELETE_COLUMN, DUBIOUS_COLUMN}
    ]
    empty = set(candidates)
    for row in rows:
        for name in list(empty):
            if _clean_cell(row.get(name)):
                empty.discard(name)
    return sorted(empty)


def clean_dataframe(
    source_path: str | Path,
    *,
    country: str = "",
    keep_schema: Sequence[str] = KEEP_SCHEMA_ORDER,
) -> CleaningOutcome:
    """Run the full generic cleaning pass over one country dataframe."""
    header, rows = load_source(source_path)
    resolved_country = country or next(
        (_clean_cell(row.get("country")) for row in rows if _clean_cell(row.get("country"))),
        "",
    )
    if is_pseudo_country(resolved_country):
        raise CleaningError(
            f"{source_path}: {resolved_country!r} is a storage pseudo-country, "
            "not a real EP24 country; it must not be cleaned as a country"
        )

    obsolete = resolve_obsolete_columns(header)
    empty_columns = find_always_empty_columns(header, rows)

    ids: dict[str, int] = {}
    for row in rows:
        key = _clean_cell(row.get("video_id"))
        if key:
            ids[key] = ids.get(key, 0) + 1
    duplicate_ids = frozenset(key for key, seen in ids.items() if seen > 1)

    decisions: list[RowDecision] = []
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        decision = compute_cleaning_result(row, duplicate_ids=duplicate_ids)
        video_id = _clean_cell(row.get("video_id"))
        if not decision.video_id:
            decision.video_id = video_id
        if not decision.country:
            decision.country = _clean_cell(row.get("country"))
        if not decision.profile:
            decision.profile = _clean_cell(row.get("author_username"))
        if not decision.platform:
            decision.platform = _clean_cell(row.get("source_type"))
        decisions.append(decision)
        enriched_rows.append(_apply_decision(row, decision))

    if len(enriched_rows) != len(rows):
        raise CleaningError("cleaning changed the row count; refusing to write output")

    legacy_removed = obsolete.resolved
    cleaned_columns = [
        name
        for name in (*header, *CLEANING_FIELDS)
        if name not in set(legacy_removed) and name not in set(RERUN_COLUMNS)
    ]
    if "video_id" not in cleaned_columns:
        raise CleaningError("video_id missing from the cleaned column set")

    return CleaningOutcome(
        country=resolved_country,
        source_path=Path(source_path),
        source_rows=len(rows),
        source_columns=list(header),
        cleaned_columns=cleaned_columns,
        removed_columns={
            "by_spreadsheet_letters": obsolete.letters,
            "lda_family": obsolete.lda,
            "always_empty": empty_columns,
            "rerun_flags": list(RERUN_COLUMNS),
        },
        empty_columns=empty_columns,
        decisions=decisions,
        rows=enriched_rows,
    )


def reprocess_columns(keep_schema: Sequence[str] = KEEP_SCHEMA_ORDER) -> list[str]:
    """Deterministic keep-schema output order with video_id first."""
    columns = list(keep_schema)
    if VIDEO_ID_FIRST and "video_id" in columns:
        columns.remove("video_id")
        columns.insert(0, "video_id")
    return columns


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def write_outputs(
    outcome: CleaningOutcome,
    *,
    output_dir: str | Path,
    to_reprocess_dir: str | Path,
    recut_worklist: str | Path,
    keep_schema: Sequence[str] = KEEP_SCHEMA_ORDER,
) -> dict[str, Path]:
    """Write every derivative artifact. Never touches the source file."""
    output = Path(output_dir)
    slug = re.sub(r"[^a-z0-9]+", "_", outcome.country.casefold()).strip("_") or "unknown"

    cleaned_path = output / f"ep24_{slug}_cleaned.csv"
    _write_csv(cleaned_path, outcome.cleaned_columns, outcome.rows)

    decisions_path = output / f"ep24_{slug}_cleaning_decisions.csv"
    _write_csv(
        decisions_path, DECISION_FIELDS, (decision.as_row() for decision in outcome.decisions)
    )

    provenance_path = output / f"ep24_{slug}_cleaning_provenance.json"
    provenance = {
        "schema": "ep24-cleaning-provenance-v1",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "country": outcome.country,
        "source": {
            "path": outcome.source_path.name,
            "rows": outcome.source_rows,
            "columns": outcome.source_columns,
        },
        "output": {"rows": len(outcome.rows), "columns": outcome.cleaned_columns},
        "removed_columns": outcome.removed_columns,
        "media_analysis_skip_seconds": VIDEO_INITIAL_SKIP_SECONDS,
        "rules": sorted({decision.rule_id for decision in outcome.decisions if decision.rule_id}),
        "counts": outcome.counts(),
        "blank_is_not_false": True,
        "destructive": False,
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    columns = reprocess_columns(keep_schema)
    reprocess_path = Path(to_reprocess_dir) / f"ep24_{slug}.csv"
    _write_csv(reprocess_path, columns, outcome.reprocess_rows())

    _append_recut_worklist(Path(recut_worklist), outcome)

    report_path = output / f"ep24_{slug}_cleaning_report.md"
    report_path.write_text(
        build_report(
            outcome,
            reprocess_columns=columns,
            files={
                "cleaned": cleaned_path.name,
                "decisions": decisions_path.name,
                "provenance": provenance_path.name,
                "reprocess": str(reprocess_path),
                "recut_worklist": str(recut_worklist),
            },
        ),
        encoding="utf-8",
    )
    return {
        "cleaned": cleaned_path,
        "decisions": decisions_path,
        "provenance": provenance_path,
        "report": report_path,
        "reprocess": reprocess_path,
    }


RECUT_WORKLIST_COLUMNS: tuple[str, ...] = (
    "video_id",
    "country",
    "author_username",
    "source_type",
    "allas_filename",
    "video_filename",
    "cleaning_status",
    "researcher_note",
)


def _append_recut_worklist(path: Path, outcome: CleaningOutcome) -> None:
    """Aggregate cut/split cases across countries into one shared worklist."""
    recut = outcome.recut_rows()
    if not recut:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    write_header = not path.exists()
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing.add(f"{row.get('country')}|{row.get('video_id')}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RECUT_WORKLIST_COLUMNS), extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        for row in recut:
            key = f"{row.get('country')}|{row.get('video_id')}"
            if key in existing:
                continue
            existing.add(key)
            writer.writerow({name: row.get(name, "") for name in RECUT_WORKLIST_COLUMNS})


def build_report(
    outcome: CleaningOutcome,
    *,
    reprocess_columns: Sequence[str],
    files: Mapping[str, str],
) -> str:
    counts = outcome.counts()
    kept = counts.get(DECISION_KEEP, 0) + counts.get(DECISION_KEEP_WITH_CORRECTION, 0)
    excluded = counts.get(DECISION_EXCLUDE, 0)
    needs_review = sum(1 for row in outcome.rows if row.get("needs_human_review") == "true")
    lines = [
        f"# EP24 cleaning report — {outcome.country or 'unknown'}",
        "",
        "Non-destructive pass. The legacy source CSV is untouched; every source row is",
        "retained here with additive cleaning fields, and exclusions are recorded as flags.",
        "",
        "## Counts",
        "",
        f"- source rows: {outcome.source_rows}",
        f"- output rows: {len(outcome.rows)}",
        f"- rows included in reprocessing: {len(outcome.reprocess_rows())}",
        f"- rows kept: {kept}",
        f"- rows excluded from new analysis: {excluded}",
        f"- rows needing human review: {needs_review}",
        f"- rows moved to the recut worklist: {len(outcome.recut_rows())}",
        "",
        "### Decision breakdown",
        "",
    ]
    for name in sorted(counts):
        lines.append(f"- `{name}`: {counts[name]}")
    lines += [
        "",
        "## Columns",
        "",
        f"- source columns: {len(outcome.source_columns)}",
        f"- cleaned columns: {len(outcome.cleaned_columns)}",
        "",
        "### Removed by legacy spreadsheet letter",
        "",
    ]
    for letters, name in sorted(outcome.removed_columns.get("by_spreadsheet_letters", {}).items()):
        lines.append(f"- {letters} -> `{name}`")
    lines += ["", "### Removed LDA family (AH–AT)", ""]
    for name in outcome.removed_columns.get("lda_family", []):
        lines.append(f"- `{name}`")
    lines += ["", "### Always-empty legacy columns", ""]
    lines += [f"- `{name}`" for name in outcome.empty_columns] or ["- (none)"]
    lines += [
        "",
        "### Researcher rerun flags retained only in provenance",
        "",
    ]
    lines += [f"- `{name}`" for name in outcome.removed_columns.get("rerun_flags", [])]
    lines += [
        "",
        "## Reprocessing output",
        "",
        f"- file: `{files.get('reprocess', '')}`",
        f"- keep-schema columns ({len(reprocess_columns)}), `video_id` first:",
        "",
        "```",
        "\n".join(reprocess_columns),
        "```",
        "",
        "Blank researcher values remain distinct from explicit `false`; no blank was",
        "coerced. Media analysis must skip the canonical first",
        f"{VIDEO_INITIAL_SKIP_SECONDS:g} s (imported from `ep24_video.py`).",
        "",
        "## Artifacts",
        "",
    ]
    lines += [f"- `{key}`: `{value}`" for key, value in sorted(files.items())]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="private per-country CSV (never modified)")
    parser.add_argument("--country", default="", help="country name; defaults to the data")
    parser.add_argument("--output-dir", help="default: the source file's directory")
    parser.add_argument("--to-reprocess-dir", required=True)
    parser.add_argument("--recut-worklist", required=True)
    args = parser.parse_args(argv)

    source = Path(args.source)
    output_dir = Path(args.output_dir) if args.output_dir else source.parent
    outcome = clean_dataframe(source, country=args.country)
    written = write_outputs(
        outcome,
        output_dir=output_dir,
        to_reprocess_dir=args.to_reprocess_dir,
        recut_worklist=args.recut_worklist,
    )
    print(
        json.dumps(
            {
                "country": outcome.country,
                "source_rows": outcome.source_rows,
                "output_rows": len(outcome.rows),
                "reprocess_rows": len(outcome.reprocess_rows()),
                "counts": outcome.counts(),
                "written": {key: str(value) for key, value in written.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
