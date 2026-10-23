"""Shared dataframe contract for the active EP24 Roihu pipeline.

The active EP24 reprocessing corpus is a researcher-recorded feed dataset.
Every stage is additive: source metadata and all upstream analysis columns are
carried forward unchanged, and only new stage columns are appended/updated.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ep24_schema import LEGACY_ALIASES, REQUIRED_MEDIA_COLUMNS, value

SOURCE_METADATA_COLUMNS: tuple[str, ...] = ()
RESEARCHER_COLUMNS = (
    "political_preference",
    "entities",
    "themes",
    "researcher_note",
)
MODEL_PREFIXES = (
    "whisper_", "asr_", "ocr_", "frame_", "video_", "summary_", "laclau_",
    "formula_of_populism_", "dna_", "sna_", "rdf_", "codebook_", "memory_", "rag_",
    "entity_normalization_", "theme_normalization_", "context_", "preprocess_",
)


def load_cumulative_csv(path: str | Path, *, require_canonical: bool = True) -> pd.DataFrame:
    """Load a stage CSV losslessly and expose only necessary media aliases."""
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
        missing = [c for c in REQUIRED_MEDIA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "EP24 media input is missing required columns: " + ", ".join(missing)
            )
    return df


def assert_source_metadata_preserved(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    mutable_columns: Iterable[str] = (),
) -> None:
    """Fail if a stage drops or mutates incoming columns except explicit aggregates."""
    mutable = set(mutable_columns)
    missing = [column for column in before.columns if column not in after.columns]
    if missing:
        raise AssertionError(f"stage dropped incoming columns: {missing}")
    if len(before) != len(after):
        raise AssertionError(
            f"stage changed row count: {len(before)} -> {len(after)}"
        )
    for column in before.columns:
        if column in mutable:
            continue
        left = before[column].astype(str).tolist()
        right = after[column].astype(str).tolist()
        if left != right:
            raise AssertionError(f"stage mutated incoming column: {column}")


def write_cumulative_csv(
    before: pd.DataFrame,
    after: pd.DataFrame,
    path: str | Path,
    *,
    mutable_columns: Iterable[str] = (),
) -> None:
    """Write an additive stage output after verifying the source contract."""
    assert_source_metadata_preserved(before, after, mutable_columns=mutable_columns)
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
        elif include_model_fields and (
            column.startswith(MODEL_PREFIXES)
            or column in {
                "topics", "positive", "neutral", "negative",
                "summary_analysis", "metadata", "SCROLL", "SCROLL_SECONDS",
            }
        ):
            model_lines.append(line)
        else:
            source_lines.append(line)
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


_HEPP24_ID = re.compile(
    r"^HEPP24-([A-Z]{2})-([A-Z0-9]+)-([A-Z0-9]+)-(\d{8})-([A-Za-z0-9._-]+)$",
    re.IGNORECASE,
)


def recover_media_identity(row: pd.Series) -> tuple[str, str]:
    """Recover missing fields from the original composite Mongo source ID.

    Historical Step-0 identities have the form
    VIDEO_ID|Country|https://a3s.fi/swift/v1/.../VIDEO_ID.mp4.
    A hash-only _storage_id cannot be inverted. Existing fields always win;
    conflicting IDs or countries must not silently relabel a source.
    """
    video_id = value(row, "video_id")
    allas = value(row, "allas_filename")
    parts = value(row, "_storage_id").split("|")
    if len(parts) < 3:
        return video_id, allas
    candidate_id, candidate_country, candidate_url = parts[0], parts[1], "|".join(parts[2:])
    match = _HEPP24_ID.fullmatch(candidate_id)
    if not match or not candidate_url.startswith(("https://", "http://")):
        return video_id, allas
    country_code = match.group(1).casefold()
    names = {"fi": "finland", "pl": "poland", "pt": "portugal",
             "de": "germany", "es": "spain", "hu": "hungary",
             "hr": "croatia", "fr": "france", "bg": "bulgaria",
             "se": "sweden"}
    allowed = {country_code, names.get(country_code, "")}
    existing_country = value(row, "country").casefold()
    if candidate_country.strip().casefold() not in allowed or (
        existing_country and existing_country not in allowed
    ):
        return video_id, allas
    from urllib.parse import urlsplit
    url = urlsplit(candidate_url)
    expected = (
        f"/HEPP24/{match.group(1).upper()}/{match.group(2).upper()}/"
        f"{match.group(3).upper()}/{match.group(4)}/{candidate_id}.mp4"
    )
    if not url.netloc or not url.path.endswith(expected) or url.username or url.password:
        return video_id, allas
    if video_id and video_id != candidate_id:
        return video_id, allas
    if allas and allas != candidate_url:
        return video_id, allas
    return video_id or candidate_id, allas or candidate_url


def recover_allas_filename(row: pd.Series) -> str:
    """Recover the documented HEPP24 Swift object URL from a structured ID.

    Unstructured video IDs are not sufficient to identify a unique object.
    Never invent a storage location for those records.
    """
    video_id, existing = recover_media_identity(row)
    if existing:
        return existing
    match = _HEPP24_ID.fullmatch(video_id)
    if not match:
        return ""
    country, account, platform, date, _sequence = match.groups()
    supplied_country = value(row, "country").strip().casefold()
    country_names = {
        "fi": "finland", "pl": "poland", "pt": "portugal",
        "de": "germany", "es": "spain", "hu": "hungary",
        "hr": "croatia", "fr": "france", "bg": "bulgaria",
        "se": "sweden",
    }
    if supplied_country and supplied_country not in {
        country.casefold(), country_names.get(country.casefold(), "")
    }:
        return ""
    base = os.getenv(
        "LACLAUGPT_ALLAS_HEPP24_BASE_URL", "https://a3s.fi/swift/v1"
    ).rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise ValueError("LACLAUGPT_ALLAS_HEPP24_BASE_URL must be HTTP(S)")
    return (
        f"{base}/HEPP24/{country.upper()}/{account.upper()}/"
        f"{platform.upper()}/{date}/{video_id}.mp4"
    )


def media_key(row: pd.Series) -> str:
    """Resolve a canonical URL/key, recovering only validated HEPP24 IDs."""
    key = recover_allas_filename(row)
    if not key:
        raise ValueError(
            "EP24 row has no resolvable allas_filename for video_id="
            + repr(value(row, "video_id"))
        )
    return key


def local_media_path(row: pd.Series, root: str | Path | None = None) -> Path:
    """Resolve a staged local copy without treating HTTP URLs as paths."""
    key = media_key(row)
    base = Path(root or os.getenv("LACLAUGPT_ALLAS_LOCAL_ROOT", "./Allas"))
    if key.startswith(("https://", "http://")):
        from urllib.parse import urlsplit
        from hashlib import sha256

        url = urlsplit(key)
        # Keep URL data under a deterministic, safe local filename.
        digest = sha256(key.encode("utf-8")).hexdigest()[:16]
        return base / "_http" / digest / Path(url.path).name
    direct = Path(key)
    if direct.exists():
        return direct
    return base / key.lstrip("/")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add absent output columns while preserving every existing column/value."""
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df
