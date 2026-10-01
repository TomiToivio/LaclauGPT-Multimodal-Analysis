"""Auditable EP24 legacy-data cleaner for issue #35.

This module contains no private EP24 rows. It provides reusable rules that operate on
Pandas DataFrames supplied by LaclauGPT-Private and writes derivative outputs only.

Key contracts:
* source DataFrames are never mutated in place;
* blank researcher values remain distinct from explicit False;
* explicit delete/dubious flags exclude rows from reprocessing;
* explicit cut/split flags route rows to the recut worklist;
* only synthetic profiles are exported for the current reprocessing target;
* the same authoritative keep-schema is used for every country;
* human-curated entity/theme fields seed canonical recyclable metadata;
* legacy model outputs are not recycled as next-round supervision.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

RULE_VERSION = "issue-35-v1"
PSEUDO_COUNTRIES = {"mobilebackup"}
TRUE_TOKENS = {"true", "1", "yes", "y", "t"}
FALSE_TOKENS = {"false", "0", "no", "n", "f"}
SYNTHETIC_TOKENS = {"synthetic", "synthetic profile", "synthetic_profile"}
TRIVIAL_NOTES = {"test"}
UNUSABLE_NOTES = {"this is filthy garbage"}

# Old machine analysis that must be regenerated rather than recycled into the new run.
LEGACY_MODEL_OUTPUTS = {
    "summary_analysis",
    "formula_of_populism_analysis",
    "topics",
    "positive",
    "neutral",
    "negative",
    "manifestoberta_predicted_class",
    "whisper_transcript",
    "whisper_language",
    "whisper_translated",
}
LEGACY_MODEL_OUTPUTS.update({f"frame_{i}" for i in range(1, 7)})
LEGACY_MODEL_OUTPUTS.update({f"ocr_{i}" for i in range(1, 7)})


@dataclass(frozen=True)
class CleaningSummary:
    country: str
    source_rows: int
    reprocess_rows: int
    recut_rows: int
    excluded_delete: int
    excluded_dubious: int
    excluded_non_synthetic: int
    needs_review: int
    rule_version: str = RULE_VERSION


def _is_blank(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def explicit_bool(value: object) -> bool | None:
    """Return True/False only for explicit boolean values; blank stays None."""
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value).strip().casefold()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def spreadsheet_column_name(position: int) -> str:
    """Convert a 1-based spreadsheet position to A1-style column letters."""
    if position < 1:
        raise ValueError("position must be >= 1")
    letters = []
    while position:
        position, remainder = divmod(position - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def resolve_legacy_drop_columns(columns: Sequence[str]) -> dict[str, str]:
    """Resolve issue #35's letter-based legacy removals against the actual header."""
    targets = ["Y", "Z", "AA", "AB", "AE"] + [
        spreadsheet_column_name(i) for i in range(34, 47)  # AH:AT
    ]
    resolved: dict[str, str] = {}
    for index, name in enumerate(columns, start=1):
        letter = spreadsheet_column_name(index)
        if letter in targets:
            resolved[letter] = name
    missing = set(targets) - set(resolved)
    if missing:
        raise ValueError(
            "CSV header is too short or incompatible with legacy letter mapping; "
            f"missing positions: {sorted(missing)}"
        )
    return resolved


def read_keep_schema(path: str | Path) -> list[str]:
    """Read the single authoritative EP24 keep-schema from CSV or XLSX."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path, dtype="string")
    elif path.suffix.casefold() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, dtype="string")
    else:
        raise ValueError(f"unsupported keep-schema format: {path.suffix}")

    # The workbook is a researcher-owned schema artifact. Prefer its headers when
    # they are meaningful; otherwise accept a one-column list of field names.
    headers = [str(c).strip() for c in frame.columns if str(c).strip()]
    if len(headers) > 1:
        return headers
    if len(headers) == 1 and headers[0].casefold() not in {
        "column",
        "columns",
        "field",
        "fields",
        "name",
    }:
        return headers

    values: list[str] = []
    if not frame.empty:
        for value in frame.iloc[:, 0].tolist():
            if not _is_blank(value):
                values.append(str(value).strip())
    if not values:
        raise ValueError(f"keep-schema contains no columns: {path}")
    return values


def validate_country(country: object) -> str:
    if _is_blank(country):
        raise ValueError("country is required and must come from authoritative metadata")
    canonical = str(country).strip()
    if canonical.casefold() in PSEUDO_COUNTRIES:
        raise ValueError(f"pseudo-country rejected: {canonical}")
    return canonical


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _profile_is_synthetic(row: pd.Series) -> bool:
    flag_col = _first_existing(
        row.index,
        (
            "account_type",
            "profile_type",
            "profile_kind",
            "profile_category",
            "synthetic_profile",
            "is_synthetic",
        ),
    )
    if flag_col is None:
        # Issue #35 requires synthetic-only export, so absence must be reviewable,
        # not silently interpreted as synthetic.
        return False
    value = row.get(flag_col)
    explicit = explicit_bool(value)
    if explicit is not None:
        return explicit
    if _is_blank(value):
        return False
    return str(value).strip().casefold() in SYNTHETIC_TOKENS


def _flagged(row: pd.Series, prefix: str) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    for column in row.index:
        if column == prefix or column.startswith(prefix):
            if explicit_bool(row.get(column)) is True:
                evidence.append(column)
    return bool(evidence), evidence


def _split_labels(value: object) -> list[str]:
    if _is_blank(value):
        return []
    text = str(value).strip()
    # Human fields historically used commas, semicolons, pipes and line breaks.
    parts = re.split(r"[;,|\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def _canonical_label(label: str, aliases: Mapping[str, str] | None) -> str:
    if not aliases:
        return label
    folded = label.casefold()
    normalized_aliases = {str(k).strip().casefold(): str(v).strip() for k, v in aliases.items()}
    return normalized_aliases.get(folded, label)


def merge_human_labels(
    row: pd.Series,
    source_columns: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Merge human-curated labels, de-duplicated by canonical identity.

    Returns (pipe-delimited values, JSON provenance).
    """
    seen: set[str] = set()
    values: list[str] = []
    provenance: list[dict[str, str]] = []
    for column in source_columns:
        if column not in row.index:
            continue
        for raw in _split_labels(row.get(column)):
            canonical = _canonical_label(raw, aliases)
            key = canonical.casefold()
            if key not in seen:
                seen.add(key)
                values.append(canonical)
            provenance.append({"source": column, "raw": raw, "canonical": canonical})
    return " | ".join(values), json.dumps(provenance, ensure_ascii=False)


def normalize_researcher_note(value: object) -> tuple[str, bool, bool]:
    """Return normalized note, human-review flag, and explicit unusable-note flag."""
    if _is_blank(value):
        return "", False, False
    note = str(value).strip()
    if note.casefold() in TRIVIAL_NOTES:
        return "", False, False
    if note.casefold() in UNUSABLE_NOTES:
        return note, False, True
    # Free-text notes are contextual evidence, not automatic ground truth.
    return note, True, False


def clean_dataframe(
    source: pd.DataFrame,
    *,
    country: str,
    keep_schema: Sequence[str],
    entity_aliases: Mapping[str, str] | None = None,
    theme_aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CleaningSummary]:
    """Create reprocess, decision-log, recut and summary outputs without mutation."""
    country = validate_country(country)
    frame = source.copy(deep=True)
    original_columns = list(frame.columns)
    source_rows = len(frame)

    id_col = _first_existing(
        frame.columns, ("video_id", "new_id", "id", "document_id", "row_id")
    )
    if id_col is None:
        raise ValueError("no stable row/video identifier found")

    # Resolve the legacy spreadsheet mapping before any column removal, as required.
    resolved_legacy = resolve_legacy_drop_columns(original_columns)
    resolved_drop = set(resolved_legacy.values())

    # LDA is always obsolete, regardless of historical spreadsheet position.
    resolved_drop.update(column for column in frame.columns if column.startswith("lda_"))

    decisions: list[dict[str, object]] = []
    recut_rows: list[pd.Series] = []
    export_rows: list[pd.Series] = []
    counts = {
        "delete": 0,
        "dubious": 0,
        "non_synthetic": 0,
        "review": 0,
    }

    for _, row in frame.iterrows():
        row_id = row.get(id_col)
        delete = explicit_bool(row.get("researcher_delete_video")) is True
        dubious = explicit_bool(row.get("researcher_dubious_video")) is True
        cut, cut_fields = _flagged(row, "researcher_cut_")
        split, split_fields = _flagged(row, "researcher_split_")
        synthetic = _profile_is_synthetic(row)

        note, note_review, note_unusable = normalize_researcher_note(row.get("researcher_note"))
        status = "KEEP"
        include = True
        needs_review = False
        reasons: list[str] = []
        evidence: list[str] = []

        if delete:
            status = "EXCLUDE_FROM_NEW_ANALYSIS"
            include = False
            counts["delete"] += 1
            reasons.append("researcher_delete_video explicitly TRUE")
            evidence.append("researcher_delete_video")
        elif dubious:
            status = "DUBIOUS_REVIEW"
            include = False
            counts["dubious"] += 1
            reasons.append("researcher_dubious_video explicitly TRUE")
            evidence.append("researcher_dubious_video")
        elif cut or split:
            status = "CUT_REQUIRED" if cut and not split else "SPLIT_REQUIRED"
            include = False
            reasons.append("researcher cut/split flag explicitly TRUE")
            evidence.extend(cut_fields + split_fields)
            recut_rows.append(row.copy())
        elif note_unusable:
            status = "EXCLUDE_FROM_NEW_ANALYSIS"
            include = False
            reasons.append("researcher_note explicitly identifies unusable content")
            evidence.append("researcher_note")
        elif not synthetic:
            status = "EXCLUDE_FROM_NEW_ANALYSIS"
            include = False
            counts["non_synthetic"] += 1
            reasons.append("current reprocessing export is synthetic-profile only")

        if note_review and include:
            needs_review = True
            counts["review"] += 1
            reasons.append("non-trivial researcher_note retained as human context")
            evidence.append("researcher_note")

        entity_seed, entity_prov = merge_human_labels(
            row, ("new_entity", "researcher_new_persons"), entity_aliases
        )
        theme_seed, theme_prov = merge_human_labels(
            row, ("new_theme", "researcher_new_themes"), theme_aliases
        )

        if include:
            cleaned = row.copy()
            cleaned["entities"] = entity_seed
            cleaned["themes"] = theme_seed
            cleaned["entities_seed_provenance"] = entity_prov
            cleaned["themes_seed_provenance"] = theme_prov
            cleaned["researcher_note"] = note
            cleaned["researcher_note_source"] = "human_researcher" if note else ""
            cleaned["cleaning_status"] = status
            cleaned["cleaning_reason"] = "; ".join(reasons)
            cleaned["cleaning_rule_id"] = RULE_VERSION
            cleaned["include_in_reprocess"] = True
            cleaned["needs_human_review"] = needs_review
            export_rows.append(cleaned)

        decisions.append(
            {
                "row_id": row_id,
                "country": country,
                "decision": status,
                "include_in_reprocess": include,
                "needs_human_review": needs_review,
                "reason": "; ".join(reasons),
                "source_researcher_fields": "|".join(dict.fromkeys(evidence)),
                "researcher_note": note,
                "entities_seed": entity_seed,
                "entities_seed_provenance": entity_prov,
                "themes_seed": theme_seed,
                "themes_seed_provenance": theme_prov,
                "rule_version": RULE_VERSION,
            }
        )

    export = pd.DataFrame(export_rows)
    recut = pd.DataFrame(recut_rows)
    decision_log = pd.DataFrame(decisions)

    # Apply the one authoritative keep-schema to every country. Provenance additions
    # are explicit and stable. The historical source remains untouched.
    provenance_columns = [
        "entities_seed_provenance",
        "themes_seed_provenance",
        "researcher_note_source",
        "cleaning_status",
        "cleaning_reason",
        "cleaning_rule_id",
        "include_in_reprocess",
        "needs_human_review",
    ]
    if not export.empty:
        missing_required = [column for column in keep_schema if column not in export.columns]
        if missing_required:
            raise ValueError(
                "source does not satisfy authoritative keep-schema; missing: "
                + ", ".join(missing_required)
            )
        ordered = list(dict.fromkeys(["video_id", *keep_schema, *provenance_columns]))
        ordered = [column for column in ordered if column in export.columns]

        forbidden = resolved_drop | LEGACY_MODEL_OUTPUTS
        # Human-seeded entities/themes are deliberately reconstructed above.
        forbidden -= {"entities", "themes"}
        ordered = [column for column in ordered if column not in forbidden]
        export = export.loc[:, ordered]

    # Guardrails: no private source mutation or accidental ID rewrite.
    if list(source.columns) != original_columns or len(source) != source_rows:
        raise AssertionError("source dataframe was mutated")
    if id_col in export.columns and not export.empty:
        allowed_ids = set(source[id_col].tolist())
        if not set(export[id_col].tolist()).issubset(allowed_ids):
            raise AssertionError("row identifiers changed")

    summary = CleaningSummary(
        country=country,
        source_rows=source_rows,
        reprocess_rows=len(export),
        recut_rows=len(recut),
        excluded_delete=counts["delete"],
        excluded_dubious=counts["dubious"],
        excluded_non_synthetic=counts["non_synthetic"],
        needs_review=counts["review"],
    )
    return export, decision_log, recut, summary


def render_markdown_report(
    summary: CleaningSummary,
    *,
    source_columns: Sequence[str],
) -> str:
    resolved = resolve_legacy_drop_columns(source_columns)
    lines = [
        f"# EP24 cleaning report: {summary.country}",
        "",
        f"- Rule version: `{summary.rule_version}`",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Source rows: {summary.source_rows}",
        f"- Reprocessing rows: {summary.reprocess_rows}",
        f"- Recut/split worklist rows: {summary.recut_rows}",
        f"- Explicit researcher deletes: {summary.excluded_delete}",
        f"- Explicit researcher dubious exclusions: {summary.excluded_dubious}",
        f"- Non-synthetic exclusions: {summary.excluded_non_synthetic}",
        f"- Rows retaining non-trivial researcher notes for review/context: {summary.needs_review}",
        "",
        "## Resolved obsolete spreadsheet positions",
        "",
    ]
    lines.extend(f"- {letter}: `{name}`" for letter, name in resolved.items())
    lines += [
        "",
        "All outputs are derivatives. The historical source CSV must remain unchanged.",
        "Blank researcher fields remain unknown/unreviewed and are never coerced to False.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    source_csv: Path,
    keep_schema_path: Path,
    output_dir: Path,
    country: str,
    recut_path: Path | None = None,
) -> CleaningSummary:
    source = pd.read_csv(source_csv, keep_default_na=True)
    keep_schema = read_keep_schema(keep_schema_path)
    export, decisions, recut, summary = clean_dataframe(
        source, country=country, keep_schema=keep_schema
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = country.strip().casefold().replace(" ", "_")
    export.to_csv(output_dir / f"ep24_{slug}.csv", index=False)
    decisions.to_csv(output_dir / f"ep24_{slug}_cleaning_decisions.csv", index=False)
    (output_dir / f"ep24_{slug}_cleaning_report.md").write_text(
        render_markdown_report(summary, source_columns=list(source.columns)),
        encoding="utf-8",
    )
    (output_dir / f"ep24_{slug}_cleaning_provenance.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if recut_path is not None and not recut.empty:
        recut_path.parent.mkdir(parents=True, exist_ok=True)
        if recut_path.exists():
            existing = pd.read_csv(recut_path)
            recut = pd.concat([existing, recut], ignore_index=True)
            recut = recut.drop_duplicates()
        recut.to_csv(recut_path, index=False)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean EP24 legacy country dataframes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--keep-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--recut-worklist", type=Path)
    args = parser.parse_args()

    summary = write_outputs(
        source_csv=args.input,
        keep_schema_path=args.keep_schema,
        output_dir=args.output_dir,
        country=args.country,
        recut_path=args.recut_worklist,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
