"""EP24 Step 2: cumulative single-keyframe multimodal analysis on CSC Roihu.

Contract:
* consume the complete Step 1 dataframe and preserve every incoming field;
* analyze exactly one Step 1 keyframe at original source t=1.0 seconds;
* pass every non-empty accumulated row field to the multimodal model;
* support TikTok and Instagram without platform-specific storage assumptions;
* append one frame analysis plus provenance/status fields;
* write a schema-preserving CSV for Step 3;
* keep a small SQLite restart cache keyed by stable source identity.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import ollama
import pandas as pd

from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import (
    assert_source_metadata_preserved,
    load_cumulative_csv,
    metadata_context,
)
from ep24_schema import stable_source_id, value as ep24_value
from ep24_video import VIDEO_INITIAL_SKIP_SECONDS
from laclaugpt_quality import QUALITY_COLUMNS, merge_status, quality_decision_from_analysis

FRAME_TIMESTAMP_SECONDS = 1.0
FRAME_PROMPT_VERSION = "ep24-frame-v2-concise-20261010"
OUTPUT_COLUMNS = (
    "frame_analysis_1",
    "frame_analysis_timestamp_seconds",
    "frame_analysis_status",
    "frame_analysis_model",
    "frame_analysis_context_sha256",
    "frame_quality_status",
    "frame_quality_reason",
    *QUALITY_COLUMNS,
)
LOG_PREVIEW_CHARS = int(os.getenv("LACLAUGPT_LOG_PREVIEW_CHARS", "1200"))

Path("./logs").mkdir(exist_ok=True)
Path("./database").mkdir(exist_ok=True)

logger = logging.getLogger("roihu_frame")
if not logger.handlers:
    handler = RotatingFileHandler(
        "./logs/frame.log",
        encoding="utf-8",
        maxBytes=5_000_000,
        backupCount=10,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    import sys
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.addHandler(stream)
logger.setLevel(logging.DEBUG)

DB_PATH = Path(os.getenv("LACLAUGPT_FRAME_SQLITE", "./database/frame.db"))


def _preview(value: object, limit: int = LOG_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + f"... <{len(text)-limit} chars truncated>"


def _detect_platform(row: pd.Series) -> str:
    for column in ("source_type", "platform", "source", "site"):
        value = str(row.get(column, "") or "").strip().lower()
        if "instagram" in value or value in {"ig", "insta"}:
            return "instagram"
        if "tiktok" in value or value in {"tt"}:
            return "tiktok"
    haystack = " ".join(
        str(row.get(column, "") or "").lower()
        for column in ("allas_filename", "url", "video_url", "source_url")
    )
    if "instagram" in haystack:
        return "instagram"
    if "tiktok" in haystack:
        return "tiktok"
    return "unknown"


def _open_cache() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS frame_analysis_cache (
            source_id TEXT PRIMARY KEY,
            platform TEXT,
            author_username TEXT,
            video_id TEXT,
            frame_file TEXT,
            frame_timestamp_seconds REAL,
            model TEXT,
            context_sha256 TEXT,
            analysis TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    return connection


def _validate_keyframe(frame_file: str, timestamp: float) -> tuple[Path, tuple[int, int]]:
    if abs(float(timestamp) - FRAME_TIMESTAMP_SECONDS) > 1e-9:
        raise ValueError(
            f"Step 2 requires the Step 1 keyframe at exactly {FRAME_TIMESTAMP_SECONDS:.1f}s; "
            f"received frame_timestamp_seconds={timestamp!r}"
        )
    path = Path(frame_file)
    if not path.is_file():
        raise FileNotFoundError(f"Step 2 keyframe does not exist: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Step 2 keyframe is unreadable: {path}")
    height, width = image.shape[:2]
    logger.debug(
        "keyframe path=%s timestamp=%.1f size_bytes=%d dimensions=%dx%d",
        path,
        timestamp,
        path.stat().st_size,
        width,
        height,
    )
    return path, (width, height)


# Get the analysis from Ollama
def get_analysis(frame_file, row_context=''):
    """Analyze the single t=1.0s frame with complete cumulative row context."""
    # Social-semiotic first-pass prompt. Keep this stage descriptive and pre-discursive.
    system_prompt = """You are analyzing ONE keyframe at original t=1.0s from a
TikTok/Instagram video collected around the European Parliament elections 2024
(EP24). Perform descriptive, light multimodal social-semiotic analysis.
Describe only the image, not an imagined video narrative. Attend especially to
people, actions, political signs, slogans, party logos, flags, campaign scenes
and other visible election context. Distinguish denotation from cautious
interpretation. Treat ASR/OCR/researcher notes as fallible context, not visual
proof. Never invent identities, text, ideology, or events. Be precise, concise,
non-repetitive, and preserve legible words in their original language."""

    user_prompt = (
        "Describe the frame under these headings: (1) Visible scene and "
        "participants; (2) Political and other salient signs and their "
        "composition; (3) Legible text, graphics, interface metadata and "
        "image-text relations; (4) Uncertainty and limitations; "
        "(5) Image usability: OK, REPROCESS or DELETE with a reason. "
        "Mention gaze, gestures, framing, colour and contrasts only if "
        "meaningful. Do not infer events before or after this single frame. "
        "Prefer direct observation over background context.\\n\\n"
        f"Limited EP24 context (not primary visual evidence):\\n{row_context}"
    )
    model = ollama_model()
    logger.info("model=%s model_source=%s", model, ollama_model_source())
    logger.debug("model=%s frame_file=%s cumulative_context_chars=%d", model, frame_file, len(row_context))
    logger.debug("cumulative_context=\n%s", _preview(row_context, max(LOG_PREVIEW_CHARS, 10000)))
    frame_analysis = ''
    logger.debug(f'Processing image: {frame_file}')
    images = []
    with open(frame_file, 'rb') as f:
        raw = f.read()
        raw = base64.b64encode(raw)
        images.append(raw.decode('utf-8'))
    # Temperature 0.0 was found to be the best for this task
    options={"repeat_last_n": 64,
             "repeat_penalty": 1.1,
             "num_ctx": int(os.getenv("LACLAUGPT_FRAME_NUM_CTX", "8192")),
             "top_p": 0.9,
             "top_k": 40,
             "min_p": 0.0,
             "temperature": 0.0,
             "num_predict": 2048}
    frame_analysis = ''
    try:
        logger.debug("model_call_start model=%s image_count=%d", model, len(images))
        response = ollama.chat(model=model, 
                               messages=[
                                    {'role': 'system', 'content': system_prompt}, 
                                    {'role': 'user', 'content': user_prompt, 'images': images},
                                    ], options=options)
        frame_message = response['message']
        frame_analysis = frame_message['content']
        logger.debug("model_call_end response_chars=%d response=%s", len(frame_analysis), _preview(frame_analysis, 10000))
    except Exception as e:
        logger.exception("Frame model call failed: %s", e)
        raise
    return frame_analysis

def _row_context(row: pd.Series) -> tuple[str, str]:
    """Budget frame prompt context without changing any stored source fields.

    Retrieval hits and memory may contain other videos' entire transcripts.
    They remain in the cumulative dataframe, but cannot displace this frame's
    direct evidence from a single-frame prompt.
    """
    budget = max(1000, int(os.getenv("LACLAUGPT_FRAME_CONTEXT_MAX_CHARS", "3500")))
    priorities = (
        "video_id", "country", "source_type", "author_username",
        "source_recording", "political_preference", "entities", "themes",
        "researcher_note", "frame_timestamp_seconds", "ocr_1",
        "asr_transcript", "asr_translated", "preprocess_status",
    )
    parts = []
    remaining = budget
    excluded = {"rag_context_json", "memory_context_json",
                "codebook_context_json", "entity_normalization_json",
                "theme_normalization_json"}
    fields = (*priorities, *(str(key) for key in row.index
                             if str(key) not in priorities
                             and str(key) not in excluded
                             and not str(key).startswith(("_pipeline", "rag_", "memory_"))))
    for key in fields:
        raw = row.get(key, "")
        value = "" if raw is None else str(raw).strip()
        if not value or value.lower() == "nan":
            continue
        label = f"- {key}: "
        if remaining <= len(label) + 30:
            break
        cap = min(len(value), remaining - len(label) - 1)
        piece = label + value[:cap]
        if cap < len(value):
            piece += " [TRUNCATED FOR MODEL CONTEXT; ORIGINAL PRESERVED]"
        parts.append(piece)
        remaining -= len(piece) + 1
    context = (
        "EP24 SINGLE-FRAME EVIDENCE AND PROVENANCE. Researcher annotations "
        "are not model ground truth. OCR and ASR are upstream estimates. "
        "Full cumulative metadata, memory and RAG retrieval are retained "
        "outside this bounded prompt.\n" + "\n".join(parts)
    )
    digest = hashlib.sha256((FRAME_PROMPT_VERSION + "\\n" + context).encode("utf-8")).hexdigest()
    logger.info("frame_context_budget chars=%d limit=%d full_context_chars=%d",
                len(context), budget, len(metadata_context(row, include_model_fields=True)))
    return context, digest


def _cache_lookup(connection: sqlite3.Connection, source_id: str, context_sha256: str, model: str):
    return connection.execute(
        """
        SELECT analysis FROM frame_analysis_cache
        WHERE source_id = ? AND context_sha256 = ? AND model = ?
        """,
        (source_id, context_sha256, model),
    ).fetchone()


def _cache_store(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    platform: str,
    author_username: str,
    video_id: str,
    frame_file: str,
    model: str,
    context_sha256: str,
    analysis: str,
) -> None:
    connection.execute(
        """
        INSERT INTO frame_analysis_cache (
            source_id, platform, author_username, video_id, frame_file,
            frame_timestamp_seconds, model, context_sha256, analysis, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source_id) DO UPDATE SET
            platform=excluded.platform,
            author_username=excluded.author_username,
            video_id=excluded.video_id,
            frame_file=excluded.frame_file,
            frame_timestamp_seconds=excluded.frame_timestamp_seconds,
            model=excluded.model,
            context_sha256=excluded.context_sha256,
            analysis=excluded.analysis,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            source_id,
            platform,
            author_username,
            video_id,
            frame_file,
            FRAME_TIMESTAMP_SECONDS,
            model,
            context_sha256,
            analysis,
        ),
    )
    connection.commit()


def analyze_videos(language=None):
    """Run cumulative Step 2 frame analysis without dropping any incoming fields."""
    filename = os.getenv("LACLAUGPT_INPUT_CSV") or f"./csv/tiktok_{language}.csv"
    output = os.getenv("LACLAUGPT_OUTPUT_CSV") or filename
    canonical = bool(os.getenv("LACLAUGPT_INPUT_CSV"))

    logger.info(
        "startup input=%s output=%s model=%s sqlite=%s expected_timestamp=%.1f",
        filename,
        output,
        ollama_model(),
        DB_PATH,
        FRAME_TIMESTAMP_SECONDS,
    )
    df = load_cumulative_csv(filename, require_canonical=canonical)
    before = df.copy(deep=True)
    logger.debug("input_rows=%d input_columns=%d columns=%s", len(df), len(df.columns), list(df.columns))

    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        logger.warning("test/demo row limit active: %d of %d rows", max_rows, len(df))
        df = df.head(max_rows).copy()
        before = before.head(max_rows).copy()

    if language and "language" in df.columns:
        mask = df["language"].astype(str) == str(language)
        df = df.loc[mask].copy()
        before = before.loc[mask].copy()

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    connection = _open_cache()
    try:
        for index, row in df.iterrows():
            source_id = stable_source_id(row)
            video_id = ep24_value(row, "video_id")
            author_username = ep24_value(row, "author_username")
            platform = _detect_platform(row)
            model = ollama_model()

            logger.debug(
                "row_start index=%s source_id=%s platform=%s author=%s video_id=%s incoming_fields=%d",
                index, source_id, platform, author_username, video_id, len(row.index),
            )
            for field in row.index:
                logger.debug("row_field index=%s name=%s value=%s", index, field, _preview(row.get(field, "")))

            try:
                frame_file = str(row.get('frame_file', '')).strip()
                if not frame_file:
                    raise ValueError("Step 2 requires Step 1 field 'frame_file'")

                raw_timestamp = str(row.get("frame_timestamp_seconds", "") or "").strip()
                if not raw_timestamp:
                    raise ValueError("Step 2 requires Step 1 field 'frame_timestamp_seconds'")
                timestamp = float(raw_timestamp)
                _validate_keyframe(frame_file, timestamp)

                transcript = str(row.get("asr_transcript", row.get("whisper_transcript", "")) or "")
                logger.debug(
                    "transcript field=%s chars=%d preview=%s",
                    "asr_transcript" if "asr_transcript" in row.index else "whisper_transcript",
                    len(transcript),
                    _preview(transcript),
                )

                context, context_sha256 = _row_context(row)
                included_fields = [
                    str(column)
                    for column in row.index
                    if str(row.get(column, "") or "").strip()
                    and str(row.get(column, "") or "").strip().lower() != "nan"
                ]
                logger.debug(
                    "prompt_context fields=%d names=%s chars=%d sha256=%s",
                    len(included_fields), included_fields, len(context), context_sha256,
                )

                cached = _cache_lookup(connection, source_id, context_sha256, model)
                if cached:
                    frame_response = str(cached[0] or "")
                    status = "cached"
                    logger.debug("cache_hit source_id=%s analysis_chars=%d", source_id, len(frame_response))
                else:
                    frame_response = str(get_analysis(frame_file, context))
                    status = "ok"
                    _cache_store(
                        connection,
                        source_id=source_id,
                        platform=platform,
                        author_username=author_username,
                        video_id=video_id,
                        frame_file=frame_file,
                        model=model,
                        context_sha256=context_sha256,
                        analysis=frame_response,
                    )
                    logger.debug("cache_store source_id=%s", source_id)

                df.at[index, "frame_analysis_1"] = (
                    f"### **Frame 1 at original t={FRAME_TIMESTAMP_SECONDS:.1f} seconds**:\n"
                    f"{frame_response}\n"
                )
                df.at[index, "frame_analysis_timestamp_seconds"] = f"{FRAME_TIMESTAMP_SECONDS:.1f}"
                df.at[index, "frame_analysis_status"] = status
                df.at[index, "frame_analysis_model"] = model
                df.at[index, "frame_analysis_context_sha256"] = context_sha256
                frame_quality_status, frame_quality_reason = quality_decision_from_analysis(frame_response)
                old_status = row.get("processing_status", "OK")
                old_reason = row.get("processing_status_reason", "")
                merged_status, merged_reason = merge_status(
                    old_status,
                    frame_quality_status,
                    current_reason=old_reason,
                    new_reason=frame_quality_reason,
                )
                df.at[index, "frame_quality_status"] = frame_quality_status
                df.at[index, "frame_quality_reason"] = frame_quality_reason
                df.at[index, "processing_status"] = merged_status
                df.at[index, "processing_status_reason"] = merged_reason
                if str(old_status or "OK").upper() != merged_status:
                    logger.info(
                        "[QUALITY] %s %s -> %s: %s",
                        source_id, str(old_status or "OK").upper(), merged_status, merged_reason,
                    )
                logger.debug(
                    "row_written index=%s fields=%s status=%s",
                    index, list(OUTPUT_COLUMNS), status,
                )
            except Exception as exc:
                df.at[index, "frame_analysis_status"] = "error"
                old_status = row.get("processing_status", "OK")
                old_reason = row.get("processing_status_reason", "")
                frame_quality_status, frame_quality_reason = quality_decision_from_analysis(
                    "", failure=True, failure_reason=str(exc)
                )
                merged_status, merged_reason = merge_status(
                    old_status,
                    frame_quality_status,
                    current_reason=old_reason,
                    new_reason=frame_quality_reason,
                )
                df.at[index, "frame_quality_status"] = frame_quality_status
                df.at[index, "frame_quality_reason"] = frame_quality_reason
                df.at[index, "processing_status"] = merged_status
                df.at[index, "processing_status_reason"] = merged_reason
                logger.info("[QUALITY] %s -> %s: %s", source_id, merged_status, merged_reason)
                logger.exception(
                    "row_failed index=%s source_id=%s platform=%s video_id=%s error=%s",
                    index, source_id, platform, video_id, exc,
                )

        # Verify the stage is additive before writing. Existing columns are immutable.
        assert_source_metadata_preserved(before, df, mutable_columns=QUALITY_COLUMNS)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, encoding="utf-8")
        logger.info("output_saved path=%s rows=%d columns=%d", output, len(df), len(df.columns))
    finally:
        connection.close()
        logger.debug("sqlite_closed path=%s", DB_PATH)


# Loop through all EP2024 TikTok languages and analyze videos
# All EP2024 TikTok languages for this stage (module level: the documented
# stage contract reads it without importing or executing the stage).
# Use the country list instead of this.
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']


if __name__ == "__main__":
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        analyze_videos(None)
    else:
        for language in languages:
            analyze_videos(language)


