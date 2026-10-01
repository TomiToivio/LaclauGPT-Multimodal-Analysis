"""Shared dataframe contract for the active EP24 Roihu pipeline.

The active EP24 reprocessing corpus is a researcher-recorded feed dataset.
Every stage is additive: source metadata and all upstream analysis columns are
carried forward unchanged, and only new stage columns are appended/updated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from ep24_schema import EP24_REPROCESS_COLUMNS, LEGACY_ALIASES, value

SOURCE_METADATA_COLUMNS = EP24_REPROCESS_COLUMNS
RESEARCHER_COLUMNS = (
    "political_preference",
    "new_entity",
    "new_theme",
    "researcher_new_persons",
    "researcher_new_themes",
    "researcher_note",
)
MODEL_PREFIXES = (
    "whisper_", "ocr_", "frame_", "video_", "summary_", "laclau_",
    "formula_of_populism_", "dna_", "sna_", "rdf_", "codebook_", "memory_",
)


def load_cumulative_csv(path: str | Path, *, require_canonical: bool = True) -> pd.DataFrame:
    """Load a stage CSV losslessly and expose canonical aliases for old files."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for canonical, aliases in LEGACY_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = df[alias].astype(str)
                break
    if require_canonical:
        missing = [c for c in EP24_REPROCESS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "EP24 reprocessing input is missing canonical columns: "
                + ", ".join(missing)
            )
    return df


def assert_source_metadata_preserved(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fail fast if a stage drops or mutates canonical researcher-feed metadata."""
    missing = [c for c in EP24_REPROCESS_COLUMNS if c not in after.columns]
    if missing:
        raise AssertionError(f"stage dropped canonical EP24 columns: {missing}")
    for column in EP24_REPROCESS_COLUMNS:
        left = before[column].astype(str).tolist()
        right = after[column].astype(str).tolist()
        if left != right:
            raise AssertionError(f"stage mutated canonical EP24 column: {column}")


def write_cumulative_csv(
    before: pd.DataFrame,
    after: pd.DataFrame,
    path: str | Path,
) -> None:
    """Write an additive stage output after verifying the source contract."""
    assert_source_metadata_preserved(before, after)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    after.to_csv(path, index=False, encoding="utf-8")


def stage_input(default: str | Path) -> Path:
    """Resolve the active stage input, allowing sbatch to point at a country CSV."""
    return Path(os.getenv("LACLAUGPT_INPUT_CSV", str(default)))


def stage_output(default: str | Path) -> Path:
    """Resolve the active stage output without ever overwriting the source by accident."""
    return Path(os.getenv("LACLAUGPT_OUTPUT_CSV", str(default)))


def metadata_context(row: pd.Series, *, include_model_fields: bool = True) -> str:
    """Render cumulative context with explicit provenance classes for prompts."""
    source_lines: list[str] = []
    researcher_lines: list[str] = []
    model_lines: list[str] = []
    for column in row.index:
        raw = row.get(column, "")
        text = "" if raw is None else str(raw).strip()
        if not text or text.lower() == "nan":
            continue
        line = f"- {column}: {text}"
        if column in RESEARCHER_COLUMNS:
            researcher_lines.append(line)
        elif column in SOURCE_METADATA_COLUMNS:
            source_lines.append(line)
        elif include_model_fields and (
            column.startswith(MODEL_PREFIXES)
            or column in {
                "entities", "themes", "topics", "positive", "neutral", "negative",
                "summary_analysis", "metadata", "SCROLL", "SCROLL_SECONDS",
            }
        ):
            model_lines.append(line)
    parts = [
        "EP24 SOURCE METADATA (recorded/split researcher feed; factual source context):",
        *(source_lines or ["- <none>"]),
        "",
        "RESEARCHER ANNOTATION (human-provided, not model ground truth):",
        *(researcher_lines or ["- <none>"]),
    ]
    if include_model_fields:
        parts.extend([
            "",
            "UPSTREAM MODEL / ENRICHMENT CONTEXT (derived, not human ground truth):",
            *(model_lines or ["- <none>"]),
        ])
    return "\n".join(parts)


def media_key(row: pd.Series) -> str:
    """Canonical media key. allas_filename is authoritative for new EP24."""
    key = value(row, "allas_filename")
    if not key:
        raise ValueError("EP24 row has no allas_filename")
    return key


def local_media_path(row: pd.Series, root: str | Path | None = None) -> Path:
    """Resolve a staged local copy from allas_filename without rebuilding scraper paths."""
    key = media_key(row)
    direct = Path(key)
    if direct.exists():
        return direct
    base = Path(root or os.getenv("LACLAUGPT_ALLAS_LOCAL_ROOT", "./Allas"))
    return base / key


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add absent output columns while preserving every existing column/value."""
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df
