#!/usr/bin/env python3
"""Non-destructive cleaning/export framework for legacy EP24 per-country dataframes.

Implements the living agent prompt in
TomiToivio/LaclauGPT-Multimodal-Analysis#35: a careful, auditable cleaning
workflow for the legacy EP24 per-country CSV dataframes (for example
``ep24_finland_with_researcher_notes.csv``) that live in
``LaclauGPT-Private/analysis/ep24_reprocess/data/by_country``.

See ``docs/EP24_LEGACY_CLEANING.md`` for the rule catalogue this module
implements and for how new rules from Tomi should be added as the issue
evolves. The core guarantees, repeated here because they are safety
invariants and not just documentation:

- the private source CSV is never opened for writing and is never mutated;
- the cleaned derivative keeps every legacy row and every legacy column
  except an explicit, reported set of confirmed-dead legacy columns;
- blank researcher-note values are never coerced to ``False``;
- corrections retain both the original legacy value and the researcher
  correction, never silently replacing one with the other;
- "excluded from the new synthetic-profile analysis" is an additive flag
  (``include_in_reprocess``), never a physical row deletion, in the cleaned
  derivative. A separate, explicitly-generated reprocessing export performs
  the physical row selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

RULESET_VERSION = "ep24-legacy-clean-v1"

# "mobilebackup" leaked into the per-country file split from the historical
# storage/path structure. It is not a country and must never be processed as
# one (TomiToivio/LaclauGPT-Multimodal-Analysis#35 comment, 2024).
PSEUDO_COUNTRIES = {"mobilebackup"}

# political_preference is the *profile's* supposed political orientation as
# used by the researcher during digital ethnography, never the politics of a
# given video. Preserve the raw value; only normalize casing/whitespace.
POLITICAL_PREFERENCES = ("Centre right", "Red-green", "Far right")
_POLITICAL_PREFERENCE_BY_KEY = {value.casefold(): value for value in POLITICAL_PREFERENCES}

LDA_COLUMN_RE = re.compile(r"^lda_", re.IGNORECASE)

# Legacy spreadsheet positions confirmed obsolete by Tomi: Y/Z/AA/AB/AE are
# near-empty remnants of older dashboard versions; AH-AT is the old Gensim LDA
# family. These are *spreadsheet letters*, resolved against the real CSV
# header at run time -- never a raw positional drop.
_EXPLICIT_REMOVE_LETTERS = ("Y", "Z", "AA", "AB", "AE")


def _letter_to_index(letters: str) -> int:
    """0-based column index for a spreadsheet column letter, e.g. ``AA`` -> 26."""
    letters = letters.strip().upper()
    index = 0
    for char in letters:
        if not ("A" <= char <= "Z"):
            raise ValueError(f"invalid spreadsheet column letters: {letters!r}")
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _index_to_letter(index: int) -> str:
    """Inverse of :func:`_letter_to_index`, used only to build ranges safely."""
    index += 1
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _letter_range(start: str, end: str) -> tuple[str, ...]:
    first, last = _letter_to_index(start), _letter_to_index(end)
    return tuple(_index_to_letter(i) for i in range(first, last + 1))


# The old Gensim LDA family (lda_topic, lda_15_topic, lda_country_topic, ...)
# as it appears in the legacy dashboard, spreadsheet columns AH-AT inclusive.
LEGACY_REMOVE_LETTERS = tuple(_EXPLICIT_REMOVE_LETTERS) + _letter_range("AH", "AT")

DECISION_KEEP = "KEEP"
DECISION_KEEP_WITH_CORRECTION = "KEEP_WITH_CORRECTION"
DECISION_EXCLUDE = "EXCLUDE_FROM_NEW_ANALYSIS"
DECISION_DUBIOUS_REVIEW = "DUBIOUS_REVIEW"
DECISION_SPLIT_REQUIRED = "SPLIT_REQUIRED"
DECISION_CUT_REQUIRED = "CUT_REQUIRED"
DECISION_WRONG_LANGUAGE = "WRONG_LANGUAGE"
DECISION_RERUN_REQUIRED = "RERUN_REQUIRED"
DECISION_DUPLICATE_REVIEW = "DUPLICATE_REVIEW"
DECISION_UNKNOWN = "UNKNOWN"

_REVIEW_DECISIONS = {
    DECISION_DUBIOUS_REVIEW,
    DECISION_SPLIT_REQUIRED,
    DECISION_CUT_REQUIRED,
    DECISION_WRONG_LANGUAGE,
    DECISION_DUPLICATE_REVIEW,
    DECISION_UNKNOWN,
}

# Candidate researcher-note column names per decision trigger. Field names in
# the real private CSVs have not all been confirmed yet; this list is
# intentionally best-effort and documented in the cleaning report so missing
# or renamed columns are visible rather than silently ignored. researcher_rerun_*
# is deliberately excluded from automatic decisions: Tomi's rule update says
# those fields are obsolete for the new run and must not drive any decision.
_FLAG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "delete": ("researcher_delete_video",),
    "dubious": ("researcher_dubious_video",),
    "split": ("researcher_split_video", "researcher_split_required"),
    "cut": ("researcher_cut_video", "researcher_cut_required"),
    "wrong_language": ("researcher_wrong_language", "researcher_language_wrong"),
    "duplicate": ("researcher_duplicate_video", "researcher_duplicate"),
}

_CORRECTION_CANDIDATES = ("new_entity", "new_theme", "researcher_new_persons", "researcher_new_themes")

_ROW_ID_CANDIDATES = ("document_id", "video_id", "id", "filename", "source_recording")
_PROFILE_CANDIDATES = ("profile", "account_handle", "profile_id", "account_name")
_PLATFORM_CANDIDATES = ("platform",)
_ACCOUNT_TYPE_CANDIDATES = ("account_type",)

# Source/order metadata that must survive into the cleaned dataframe and the
# reprocessing export untouched (Tomi's rule update: useful for studying feed
# ordering / recommendation dynamics over time).
ORDER_METADATA_COLUMNS = (
    "source_type",
    "source_recording",
    "sequence_number",
    "recording_date",
    "recording_datetime",
)


def is_pseudo_country(value: Any) -> bool:
    """True for storage-path leftovers such as ``mobilebackup``, not real countries."""
    return str(value or "").strip().casefold() in PSEUDO_COUNTRIES


def tri_state(value: Any) -> str:
    """Return ``"TRUE"``, ``"FALSE"``, or ``""`` -- never coerce blank to false.

    Blank/missing means "not reviewed or not stated", which is semantically
    different from an explicit researcher ``FALSE``. Anything that isn't
    blank and isn't a recognised boolean spelling is returned unchanged
    (stripped) so an unexpected researcher note stays visible instead of
    being forced into TRUE/FALSE.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""
    if text.casefold() in {"true", "1", "yes", "y"}:
        return "TRUE"
    if text.casefold() in {"false", "0", "no", "n"}:
        return "FALSE"
    return text


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def split_values(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts = re.split(r"[,;\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def canonical_political_preference(value: Any) -> str:
    """Normalize casing/whitespace only; unrecognised/blank values stay as-is.

    Returns ``""`` for blank input and the raw (stripped) text for values that
    do not match one of the three known profile preferences, so an unknown
    value is never silently mapped onto a wrong canonical bucket.
    """
    text = _clean_text(value)
    if not text:
        return ""
    return _POLITICAL_PREFERENCE_BY_KEY.get(text.casefold(), text)


def resolve_legacy_removed_columns(header: list[str]) -> dict[str, str | None]:
    """Map each confirmed-obsolete spreadsheet letter to the real header name.

    Returns ``None`` for a letter past the end of the header (header shorter
    than expected) instead of raising, so callers can report a mismatch
    rather than silently deleting the wrong column.
    """
    resolved: dict[str, str | None] = {}
    for letters in LEGACY_REMOVE_LETTERS:
        index = _letter_to_index(letters)
        resolved[letters] = header[index] if index < len(header) else None
    return resolved


def find_lda_columns(header: Iterable[str]) -> list[str]:
    return [name for name in header if LDA_COLUMN_RE.match(str(name))]


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _row_identifier(row: pd.Series, row_id_column: str | None, position: int) -> str:
    if row_id_column:
        value = _clean_text(row.get(row_id_column))
        if value:
            return value
    return f"row-{position}"


@dataclass
class CleaningDecision:
    row_id: str
    country: str
    profile: str
    platform: str
    decision: str
    reason: str
    source_fields: str
    original_value: str
    cleaned_value: str
    human_or_derived: str
    confidence: str
    rule_id: str
    processed_at: str
    needs_human_review: bool = False

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        return out


def _decide_flags(row: pd.Series, df_columns: set[str]) -> tuple[str, list[str], list[str]]:
    """Return (trigger_name_or_empty, reasons, matched_source_fields)."""
    for trigger in ("delete", "dubious", "split", "cut", "wrong_language", "duplicate"):
        for candidate in _FLAG_CANDIDATES[trigger]:
            if candidate not in df_columns:
                continue
            state = tri_state(row.get(candidate))
            if state == "TRUE":
                return trigger, [f"{candidate}=TRUE"], [candidate]
            if state and state not in {"TRUE", "FALSE"}:
                # Ambiguous researcher text: must stay ambiguous, not forced.
                return "ambiguous", [f"{candidate}={state!r} (unrecognised boolean text)"], [candidate]
    return "", [], []


def decide_row(
    row: pd.Series,
    *,
    df_columns: set[str],
    country: str,
    row_id: str,
    profile_column: str | None,
    platform_column: str | None,
    processed_at: str,
    rule_id: str = RULESET_VERSION,
) -> CleaningDecision:
    trigger, reasons, matched_fields = _decide_flags(row, df_columns)

    correction_fields = [name for name in _CORRECTION_CANDIDATES if name in df_columns and split_values(row.get(name))]

    if trigger == "delete":
        decision, needs_review = DECISION_EXCLUDE, False
    elif trigger == "dubious":
        decision, needs_review = DECISION_EXCLUDE, False
    elif trigger == "split":
        decision, needs_review = DECISION_SPLIT_REQUIRED, True
    elif trigger == "cut":
        decision, needs_review = DECISION_CUT_REQUIRED, True
    elif trigger == "wrong_language":
        decision, needs_review = DECISION_WRONG_LANGUAGE, True
    elif trigger == "duplicate":
        decision, needs_review = DECISION_DUPLICATE_REVIEW, True
    elif trigger == "ambiguous":
        decision, needs_review = DECISION_DUBIOUS_REVIEW, True
    elif correction_fields:
        decision, needs_review = DECISION_KEEP_WITH_CORRECTION, False
        reasons = [f"{name} present" for name in correction_fields]
        matched_fields = correction_fields
    else:
        decision, needs_review = DECISION_KEEP, False

    return CleaningDecision(
        row_id=row_id,
        country=country,
        profile=_clean_text(row.get(profile_column)) if profile_column else "",
        platform=_clean_text(row.get(platform_column)) if platform_column else "",
        decision=decision,
        reason="; ".join(reasons) if reasons else "no researcher-note trigger matched",
        source_fields=",".join(matched_fields),
        original_value="",
        cleaned_value="",
        human_or_derived="human" if matched_fields else "derived",
        confidence="ambiguous" if trigger == "ambiguous" else "explicit" if matched_fields else "default",
        rule_id=rule_id,
        processed_at=processed_at,
        needs_human_review=needs_review,
    )


def build_seed_values(row: pd.Series, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """De-duplicate (case-insensitive) union of researcher-provided labels.

    Returns ``(values, provenance_fields)`` where ``provenance_fields`` lists
    which of ``columns`` actually contributed at least one value.
    """
    seen: dict[str, str] = {}
    provenance: list[str] = []
    for column in columns:
        values = split_values(row.get(column))
        if values:
            provenance.append(column)
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen[key] = value
    return list(seen.values()), provenance


def clean_dataframe(df: pd.DataFrame, *, country: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the additive cleaned derivative and the row-level decision log.

    Never mutates ``df``. Returns ``(cleaned_df, decisions_df, report)``.
    """
    if is_pseudo_country(country):
        raise ValueError(f"{country!r} is a storage-path artifact, not a country; refusing to clean it")

    header = list(df.columns)
    removed_letter_map = resolve_legacy_removed_columns(header)
    removed_legacy_columns = sorted({name for name in removed_letter_map.values() if name})
    removed_lda_columns = sorted(set(find_lda_columns(header)) - set(removed_legacy_columns))
    all_removed = sorted(set(removed_legacy_columns) | set(removed_lda_columns))

    cleaned = df.drop(columns=all_removed, errors="ignore").copy()

    row_id_column = _first_present(df, _ROW_ID_CANDIDATES)
    profile_column = _first_present(df, _PROFILE_CANDIDATES)
    platform_column = _first_present(df, _PLATFORM_CANDIDATES)
    account_type_column = _first_present(df, _ACCOUNT_TYPE_CANDIDATES)
    political_column = "political_preference" if "political_preference" in df.columns else None

    df_columns = set(df.columns)
    processed_at = datetime.now(timezone.utc).isoformat()

    decisions: list[CleaningDecision] = []
    cleaning_status: list[str] = []
    cleaning_reason: list[str] = []
    cleaning_rule_id: list[str] = []
    cleaning_notes_md: list[str] = []
    include_in_reprocess: list[bool] = []
    needs_human_review: list[bool] = []
    entities_col: list[str] = []
    entities_provenance_col: list[str] = []
    themes_col: list[str] = []
    themes_provenance_col: list[str] = []
    political_canonical_col: list[str] = []

    organic_excluded = 0
    missing_account_type = account_type_column is None

    for position, (_, row) in enumerate(df.iterrows()):
        row_id = _row_identifier(row, row_id_column, position)
        decision = decide_row(
            row,
            df_columns=df_columns,
            country=country,
            row_id=row_id,
            profile_column=profile_column,
            platform_column=platform_column,
            processed_at=processed_at,
        )
        decisions.append(decision)
        cleaning_status.append(decision.decision)
        cleaning_reason.append(decision.reason)
        cleaning_rule_id.append(decision.rule_id)
        cleaning_notes_md.append(f"- **{decision.decision}**: {decision.reason}")
        needs_human_review.append(decision.needs_human_review)

        entities, entities_provenance = build_seed_values(row, ("new_entity", "researcher_new_persons"))
        themes, themes_provenance = build_seed_values(row, ("new_theme", "researcher_new_themes"))
        entities_col.append(", ".join(entities))
        entities_provenance_col.append(",".join(entities_provenance))
        themes_col.append(", ".join(themes))
        themes_provenance_col.append(",".join(themes_provenance))

        political_canonical_col.append(canonical_political_preference(row.get(political_column)) if political_column else "")

        is_synthetic = True
        if account_type_column is not None:
            is_synthetic = _clean_text(row.get(account_type_column)).casefold() == "synthetic"
            if not is_synthetic:
                organic_excluded += 1
        include_in_reprocess.append(
            decision.decision in (DECISION_KEEP, DECISION_KEEP_WITH_CORRECTION) and is_synthetic
        )

    cleaned["row_id"] = [d.row_id for d in decisions]
    cleaned["cleaning_status"] = cleaning_status
    cleaned["cleaning_reason"] = cleaning_reason
    cleaned["cleaning_rule_id"] = cleaning_rule_id
    cleaned["cleaning_notes_md"] = cleaning_notes_md
    cleaned["include_in_reprocess"] = include_in_reprocess
    cleaned["needs_human_review"] = needs_human_review
    cleaned["entities"] = entities_col
    cleaned["entities_seed_provenance"] = entities_provenance_col
    cleaned["themes"] = themes_col
    cleaned["themes_seed_provenance"] = themes_provenance_col
    if political_column:
        cleaned["political_preference_canonical"] = political_canonical_col
    cleaned["cleaning_processed_at"] = processed_at
    cleaned["cleaning_ruleset_version"] = RULESET_VERSION

    decisions_df = pd.DataFrame([d.as_dict() for d in decisions])

    status_counts = pd.Series(cleaning_status).value_counts().to_dict()
    report = {
        "ruleset_version": RULESET_VERSION,
        "country": country,
        "processed_at": processed_at,
        "source_row_count": int(len(df)),
        "output_row_count": int(len(cleaned)),
        "status_counts": status_counts,
        "organic_profiles_excluded_from_reprocess": organic_excluded,
        "account_type_column_missing": missing_account_type,
        "include_in_reprocess_count": int(sum(include_in_reprocess)),
        "needs_human_review_count": int(sum(needs_human_review)),
        "removed_legacy_position_columns": removed_legacy_columns,
        "removed_legacy_position_letters": {k: v for k, v in removed_letter_map.items() if v},
        "unresolved_legacy_position_letters": [k for k, v in removed_letter_map.items() if v is None],
        "removed_lda_columns": removed_lda_columns,
        "row_id_column_used": row_id_column,
        "profile_column_used": profile_column,
        "platform_column_used": platform_column,
        "account_type_column_used": account_type_column,
    }

    _assert_non_destructive(df, cleaned, all_removed)
    return cleaned, decisions_df, report


def _assert_non_destructive(source_df: pd.DataFrame, cleaned_df: pd.DataFrame, removed_columns: list[str]) -> None:
    assert len(cleaned_df) == len(source_df), (
        "cleaned derivative must keep every legacy row; use the separate reprocess export "
        "for physical row selection"
    )
    expected_legacy = set(source_df.columns) - set(removed_columns)
    missing = expected_legacy - set(cleaned_df.columns)
    assert not missing, f"cleaning must not silently drop legacy columns: {sorted(missing)}"


def build_reprocess_export(cleaned_df: pd.DataFrame, *, keep_schema: list[str] | None = None) -> pd.DataFrame:
    """Physically select rows flagged ``include_in_reprocess`` for the new pipeline.

    This is the only place rows are physically dropped; the cleaned
    derivative itself always retains every legacy row.
    """
    selected = cleaned_df[cleaned_df["include_in_reprocess"]].copy()
    if keep_schema:
        present = [name for name in keep_schema if name in selected.columns]
        selected = selected[present]
    return selected


def load_keep_schema(path: str | Path) -> list[str]:
    """Load an authoritative ordered column list from JSON array or CSV/TXT.

    The authoritative keep-schema is maintained as a private ``.xlsx``
    workbook (see ``docs/EP24_LEGACY_CLEANING.md``); export its header row to
    JSON or CSV/TXT (one column name per line, or a single header row) for
    this loader rather than adding an xlsx parsing dependency here.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array of column names")
        return [str(item) for item in payload]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and ("," in lines[0]):
        return [item.strip() for item in lines[0].split(",") if item.strip()]
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# EP24 legacy cleaning report: {report['country']}",
        "",
        f"- Ruleset version: `{report['ruleset_version']}`",
        f"- Processed at: {report['processed_at']}",
        f"- Source row count: {report['source_row_count']}",
        f"- Output row count: {report['output_row_count']}",
        f"- Rows included in reprocessing export: {report['include_in_reprocess_count']}",
        f"- Rows needing human review: {report['needs_human_review_count']}",
        f"- Organic profiles excluded from reprocess target: {report['organic_profiles_excluded_from_reprocess']}",
        "",
        "## Decision counts",
        "",
    ]
    for status, count in sorted(report["status_counts"].items()):
        lines.append(f"- `{status}`: {count}")
    lines += [
        "",
        "## Columns removed (legacy spreadsheet position Y/Z/AA/AB/AE, AH-AT)",
        "",
    ]
    if report["removed_legacy_position_columns"]:
        for letter, name in sorted(report["removed_legacy_position_letters"].items()):
            lines.append(f"- `{letter}` -> `{name}`")
    else:
        lines.append("- (none resolved against this header)")
    if report["unresolved_legacy_position_letters"]:
        lines.append("")
        lines.append(
            "Unresolved letters (header shorter than expected, nothing removed for these): "
            + ", ".join(report["unresolved_legacy_position_letters"])
        )
    lines += ["", "## Columns removed (`lda_*` family)", ""]
    if report["removed_lda_columns"]:
        for name in report["removed_lda_columns"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- (none found)")
    lines += [
        "",
        "## Column resolution used",
        "",
        f"- row identifier column: `{report['row_id_column_used']}`",
        f"- profile column: `{report['profile_column_used']}`",
        f"- platform column: `{report['platform_column_used']}`",
        f"- account_type column: `{report['account_type_column_used']}`"
        + (" (missing: synthetic-only filter could not be enforced)" if report["account_type_column_missing"] else ""),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    cleaned_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    report: dict[str, Any],
    *,
    output_dir: str | Path,
    country: str,
    source_path: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if source_path is not None and Path(source_path).resolve() == output_dir.resolve():
        raise ValueError("refusing to write cleaning outputs into the source directory as the source file")
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = country.strip().lower().replace(" ", "_")

    cleaned_path = output_dir / f"ep24_{slug}_cleaned.csv"
    decisions_path = output_dir / f"ep24_{slug}_cleaning_decisions.csv"
    report_path = output_dir / f"ep24_{slug}_cleaning_report.md"
    provenance_path = output_dir / f"ep24_{slug}_cleaning_report.json"

    if source_path is not None and Path(source_path).resolve() in {
        cleaned_path.resolve(), decisions_path.resolve(), report_path.resolve(), provenance_path.resolve()
    }:
        raise ValueError("refusing to overwrite the private source CSV with a cleaning output")

    cleaned_df.to_csv(cleaned_path, index=False)
    decisions_df.to_csv(decisions_path, index=False)
    report_path.write_text(render_markdown_report(report), encoding="utf-8")
    provenance_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cleaned_csv": cleaned_path,
        "decisions_csv": decisions_path,
        "report_md": report_path,
        "report_json": provenance_path,
    }


def clean_country_file(
    source_path: str | Path,
    *,
    country: str,
    output_dir: str | Path,
    keep_schema_path: str | Path | None = None,
) -> dict[str, Path]:
    source_path = Path(source_path)
    if is_pseudo_country(source_path.stem.replace("ep24_", "").replace("_with_researcher_notes", "")):
        raise ValueError(f"{source_path} looks like a pseudo-country file (e.g. mobilebackup) and must not be cleaned")
    df = pd.read_csv(source_path)
    cleaned_df, decisions_df, report = clean_dataframe(df, country=country)
    paths = write_outputs(cleaned_df, decisions_df, report, output_dir=output_dir, country=country, source_path=source_path)

    if keep_schema_path:
        keep_schema = load_keep_schema(keep_schema_path)
        export_df = build_reprocess_export(cleaned_df, keep_schema=keep_schema)
        export_path = Path(output_dir) / f"ep24_{country.strip().lower().replace(' ', '_')}.csv"
        export_df.to_csv(export_path, index=False)
        paths["reprocess_export_csv"] = export_path

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="private per-country CSV, e.g. ep24_finland_with_researcher_notes.csv")
    parser.add_argument("--country", required=True, help="country name, e.g. Finland")
    parser.add_argument("--output-dir", required=True, help="directory for cleaned/decisions/report outputs")
    parser.add_argument("--keep-schema", default=None, help="optional JSON/CSV/TXT authoritative keep-schema column list")
    args = parser.parse_args(argv)

    paths = clean_country_file(
        args.input,
        country=args.country,
        output_dir=args.output_dir,
        keep_schema_path=args.keep_schema,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
