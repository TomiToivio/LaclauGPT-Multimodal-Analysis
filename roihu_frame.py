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
    system_prompt = f'''### System Prompt

You are performing a **multimodal social-semiotic pre-analysis** of a single frame from incoming social-media or web video a TikTok or Instagram video related to European Parliament Elections in 2024. It is recorded from a GrapheneOS phone video feed by a researcher doing digital ethnography. 

Note the political context and take into account recognizable politicians, political slogans, political symbols, country flags and political situations like voting or campaign rallies. The videos are from different countries of the European Union: Finland, Sweden, Germany, France, Spain, Portugal, Croatia, Hungary and Bulgaria. 

Use a light social-semiotic methodology inspired by Halliday/SFL, Kress & van Leeuwen, multimodal social semiotics, and structuralist attention to signs and relations. Separate observation from interpretation and mark uncertainty explicitly.

### Input
- Exactly one keyframe sampled at original source t=1.0s, immediately after the known feed-scroll artifact.
- Treat this as the deep visual/context still that complements the later whole-video narrative analysis.
- It may contain people, objects, environments, captions, subtitles, memes, screenshots, platform UI, graphics, diagrams, logos, symbols, emojis, or embedded media.
- Inspect platform/video metadata that is visibly rendered in the frame: username/handle, display name, date/time, title/caption, hashtags, subtitles, counters, labels, buttons and other interface text. Report only what is actually visible and mark uncertainty.
- You also receive other information like date the video feed was recorded, political preference of the synthetic profile of the researcher recording the video, transcript of the video etc. Focus on the visual analysis of the keyframe but you can use the other data to augment your analysis.

### Analysis categories

1. **Denotative description**
   - Describe only what is visibly present.
   - Include people without identifying unknown persons, objects, setting, actions frozen in the frame, text, graphics, interface elements, and embedded images/screens.
   - Keep in mind the European Parliament Elections 2024 context. Note any recognizable politicians, party symbols and situations like campaign rallies.
   - Distinguish observation from inference.

2. **Semiotic resources / modes**
   - Identify visible resources such as photographic image, illustration, writing, typography, colour, gesture/posture, spatial arrangement, symbols, diagrams, emojis, platform/interface elements, and image-within-image.
   - Pay attention to political symbols and party colors.
   - Note what each resource appears to contribute descriptively.

3. **Participants, processes, circumstances**
   - Participants: visible people, groups, objects, institutions represented by explicit text/logo, places, or other entities.
   - Processes: visible actions or represented processes.
   - Circumstances: visible spatial, temporal, environmental, or situational context.
   - Identify recognizable politicians. Note political roles like politician or voter and situations like voting.

4. **Composition and salience**
   - Foreground/background; centre/periphery; relative size/scale; camera distance/angle where observable; cropping; gaze/gesture direction; repetition; contrast; visual hierarchy.
   - Describe likely viewing order only when composition supports it.
   - Treat colour as a compositional resource, not as evidence of mood, ideology, nationality, or emotion unless explicit contextual evidence supports that reading.

5. **Salient signs / signifiers**
   - List especially prominent, repeated, foregrounded, or explicitly emphasized words, objects, symbols, gestures, colours, and graphic elements.
   - Think about the meaning in political context.

6. **Relations among signs**
   - Note observable juxtapositions, contrasts, pairings, repetitions, sequences implied inside the frame, part-whole relations, labels, arrows, vectors, or other relational structures.
   - Where useful, distinguish narrative/vector structures from conceptual/classificatory structures.

7. **Image–text / intermodal relations**
   - If text and image coexist, describe whether they appear redundant, complementary/extending, elaborating/anchoring, or contrasting.
   - Quote short visible text exactly when legible. Mark OCR-like uncertainty rather than guessing.

8. **Connotation, cautiously**
   - Record culturally available associations only when strongly supported by conventional signs or explicit context.
   - Keep connotation separate from denotation and offer multiple plausible readings when appropriate.
   - Never turn connotation into political/discourse analysis at this stage.

9. **Ambiguity and uncertainty**
   - List unclear identities, illegible text, ambiguous symbols, uncertain scene context, cropping limitations, or interpretations that require other frames/audio/transcript.

10. **Video metadata**
   - These videos are from TikTok and Instagram feeds: list any visible metadata.
   - List the author username of the creator of TikTok or Instagram video.
   - Also list other visible metadata like hashtags, video title, date, other visible text.

11. **Video problems**    
   - Note if the frame has problems, for an example it seems like there is no meaningful content in the screen.
   - Indicate if you think the video is OK, should be REPROCESSED or DELETED.
   - Clearly indicate if the video is `OK`, or mark it for `REPROCESS` or `DELETE`.

### Output
Produce a detailed structured description under the headings above, and include:
- **Visible platform/video metadata:** username/handle, date/time, title/caption, hashtags, subtitles, interface labels and other metadata-like text actually visible on screen.
- **Visible text transcription:** preserve exact text where legible and distinguish it from OCR/upstream transcript context.
- **Detailed scene inventory:** people, objects, setting, clothing, gestures, graphics, logos, symbols, composition and small but potentially relevant details.
- **Frame gist:** 1–3 neutral sentences.
- **Preserve for downstream analysis:** exact visible words/phrases and salient signs that later stages should receive unchanged where possible.
- **Uncertainty:** everything unclear, cropped, illegible or dependent on temporal context.
'''

    user_prompt = f'''
Analyze the provided frame using the social-semiotic pre-analysis categories above. Stay descriptive and modality-aware. Do not perform discourse or political analysis, and do not infer ideology, persuasion, populism, sentiment, or political alignment.\n\nCUMULATIVE EP24 CONTEXT:\n{row_context}\n'''
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
             "num_ctx": 8192,
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
    """Return complete provenance-labelled row context and its reproducibility hash."""
    context = metadata_context(row, include_model_fields=True)
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
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


