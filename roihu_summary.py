import hashlib
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler

from ep24_db import country_storage
from ep24_memory import retrieve_researcher_memory
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import load_cumulative_csv, metadata_context, write_cumulative_csv
from ep24_rag import retrieve_stage_rag, upsert_stage_rag
from ep24_redis import RedisCoordinator
from ep24_schema import stable_source_id, value as ep24_value
from roihu_storage import StorageConfig
logger = logging.getLogger(__name__)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./database', exist_ok=True)
logging.basicConfig(handlers=[RotatingFileHandler('./logs/summary.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

OUTPUT_COLUMNS = ("metadata", "summary_analysis", "summary_summary_md")
DB_PATH = Path(os.getenv("LACLAUGPT_SUMMARY_SQLITE", "./database/summary.db"))
DEDICATED_MODAL_COLUMNS = (
    "frame_analysis_1",
    "ocr_1",
    "asr_transcript",
    "asr_translated",
    "whisper_transcript",
    "whisper_translated",
    "whisperResult",
    "vllm_video_analysis",
    "vllm_video_markdown_analysis",
    "metadata",
    "summary_analysis",
    "summary_summary_md",
)


def _open_cache() -> sqlite3.Connection:
    """Open the Step-4 restart cache. CSV/Mongo remain the cumulative stores."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summary_cache (
            source_id TEXT NOT NULL,
            model TEXT NOT NULL,
            context_sha256 TEXT NOT NULL,
            summary_analysis TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source_id, model, context_sha256)
        )
        """
    )
    conn.commit()
    return conn


def _cache_lookup(
    conn: sqlite3.Connection, source_id: str, model: str, context_sha256: str
) -> str | None:
    row = conn.execute(
        """SELECT summary_analysis FROM summary_cache
           WHERE source_id=? AND model=? AND context_sha256=?""",
        (source_id, model, context_sha256),
    ).fetchone()
    return None if row is None else str(row[0] or "")


def _cache_store(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    model: str,
    context_sha256: str,
    summary_analysis: str,
) -> None:
    conn.execute(
        """INSERT INTO summary_cache
           (source_id, model, context_sha256, summary_analysis, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(source_id, model, context_sha256) DO UPDATE SET
             summary_analysis=excluded.summary_analysis,
             updated_at=excluded.updated_at""",
        (
            source_id,
            model,
            context_sha256,
            summary_analysis,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _metadata_for_prompt(row):
    """Keep the full cumulative row except evidence already supplied in dedicated blocks."""
    reduced = row.drop(labels=[c for c in DEDICATED_MODAL_COLUMNS if c in row.index])
    return metadata_context(reduced, include_model_fields=True)


def _evidence_from_row(row) -> tuple[str, str, str, str]:
    metadata = _metadata_for_prompt(row)
    transcript = (
        str(row.get("asr_translated", "")).strip()
        or str(row.get("asr_transcript", "")).strip()
        or str(row.get("whisper_translated", "")).strip()
        or str(row.get("whisper_transcript", "")).strip()
        or str(row.get("whisperResult", "")).strip()
    )
    frame_parts: list[str] = []
    frame_text = str(row.get("frame_analysis_1", "")).strip()
    ocr_text = str(row.get("ocr_1", "")).strip()
    if frame_text:
        frame_parts.append(frame_text)
    if ocr_text:
        frame_parts.append("### OCR at original t=1.0s\n" + ocr_text)
    frame_analysis = "\n\n".join(frame_parts)
    video_analysis = (
        str(row.get("vllm_video_analysis", "")).strip()
        or str(row.get("vllm_video_markdown_analysis", "")).strip()
        or str(row.get("vllm_structured_output", "")).strip()
    )
    return metadata, transcript, frame_analysis, video_analysis


def _prompt_sha256(system_prompt: str, user_prompt: str, model: str) -> str:
    payload = f"{model}\n{system_prompt}\n{user_prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_storage_id(row) -> str:
    return str(row.get("_storage_id", "")).strip() or stable_source_id(row)


def _format_memory_context(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [
        "These are researcher-seeded canonical labels for normalization only.",
        "Do not treat them as evidence that the current video contains the concept.",
    ]
    for item in items:
        lines.append(
            f"- {item.get('kind', 'item')}: {item.get('label', '')} "
            f"[role={item.get('evidence_role', 'normalization_context_not_source_evidence')}]"
        )
    return "\n".join(lines)


def _format_rag_context(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [
        "These are retrieved prior-analysis records for comparison/context only.",
        "Do not treat them as direct evidence about the current video.",
    ]
    for item in items:
        excerpt = str(item.get("text", "")).replace("\n", " ")[:800]
        lines.append(
            f"- stage={item.get('stage', '')} source_record_id={item.get('source_record_id', '')}: {excerpt}"
        )
    return "\n".join(lines)


def _persist_mongo_row(storage, row, *, source_id: str, model: str, context_sha256: str) -> int:
    """Patch the complete cumulative row without replacing prior Mongo fields."""
    document = {str(k): v for k, v in row.to_dict().items() if str(k) != "_storage_id"}
    document["_storage_id"] = source_id
    document["step4_summary_provenance"] = {
        "pipeline_stage": "step_4_summary",
        "model": model,
        "context_sha256": context_sha256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return storage.patch_documents("dataframe", [document])


def _mongo_resume_summary(storage, source_id: str, *, model: str, context_sha256: str) -> str | None:
    """Use Mongo as durable resume state only when prompt/model provenance matches."""
    rows = storage.find("dataframe", {"_storage_id": source_id}, limit=1)
    if not rows:
        return None
    doc = rows[0]
    provenance = doc.get("step4_summary_provenance") or {}
    if not isinstance(provenance, dict):
        return None
    if provenance.get("model") != model or provenance.get("context_sha256") != context_sha256:
        return None
    value = str(doc.get("summary_analysis", "")).strip()
    return value or None


# Note: Only one Frame analysis from now on.
# Add video analysis
# Transcript is good to be here
# Put the rest of the dataframe columns in metadata. I mean every field the pipeline has produced so far. 
def get_llama_summary_user_prompt(
    metadata,
    transcript,
    frame_analysis,
    video_analysis,
    memory_context="",
    rag_context="",
):
    """Construct the user prompt with explicit source/context evidence boundaries."""
    user_message = f'''### User Prompt

### Input data

1. **Frame analyses**
```
{frame_analysis}
```

2. **Video analysis**
```
{video_analysis}
```

3. **Source/platform metadata**
```
{metadata}
```

4. **Speech / transcript**
```
{transcript}
```

5. **Researcher memory / normalization context (NOT source evidence)**
```
{memory_context or "<none>"}
```

6. **Retrieved prior-corpus context (NOT source evidence)**
```
{rag_context or "<none>"}
```

### Task

Integrate all available modalities into one **descriptive multimodal social-semiotic first pass**.

Treat frame descriptions, written/visible text, transcript, metadata, and temporal sequence as distinct evidence streams. Preserve disagreements between them rather than forcing a single interpretation. Metadata can provide context but must not override what is actually present in the media.

You are analyzing TikTok and Instagram videos related to the European Parliament Elections of 2024. Pay attention to multimodal political content.  
Note the political context and take into account recognizable politicians, political slogans, political symbols, country flags and political situations like voting or campaign rallies. The videos are from different countries of the European Union: Finland, Sweden, Germany, France, Spain, Portugal, Croatia, Hungary and Bulgaria.

Your task is to describe the political content carefully but don't perform in-depth discourse analysis yet. A later step will do that. 
'''
    return user_message


def get_llama_summary_system_prompt():
    """Construct the system prompt for multimodal social-semiotic pre-analysis."""
    system_prompt = '''### System Prompt

You are assisting a social-science research pipeline by creating a **Multimodal Social-Semiotic Pre-Analysis** of TikTok and Instagram videos related to the European Parliament Elections of 2024. Pay attention to multimodal political content. 

The methodological orientation is:
- social semiotics and multimodality: signs and semiotic resources make meaning across modes;
- Halliday/SFL: attend descriptively to textual organisation, represented participants/processes/circumstances, and relations between communicator/content/audience when directly observable;
- Kress & van Leeuwen: composition, salience, vectors, conceptual vs narrative visual structures, and affordances of modes;
- structuralist preparation: preserve salient signifiers, contrasts, co-occurrences, and relations for later analysis;
- a cautious denotation/connotation distinction: describe what is present first, then record only well-supported culturally available associations.

This is explicitly **before discourse analysis**. The purpose is to transform heterogeneous media into a faithful, structured account of signs and cross-modal relations that downstream LaclauGPT stages can analyze. You should however describe the political content carefully as we are analyzing European Parliament elections of 2024. Describe the political contents but don't create in-depth discourse analysis yet. 

### Epistemic rules

1. Separate **observation**, **cross-modal synthesis**, and **interpretive possibility**.
2. Never invent missing context. Mark uncertainty.
3. Preserve exact wording of important transcript/caption/visible-text signifiers where possible.
4. Treat language, image, speech, sound references present in the supplied analyses, gesture, typography, colour, spatial layout, editing/sequence, and platform/interface elements as potentially distinct semiotic resources.
5. Do not assume that metadata, captions, transcript, and imagery say the same thing.
6. Describe actor/participant roles only when directly observable or explicitly named in source material.
7. Do not infer protected traits, intentions, beliefs, ideology, party preference, political alignment, or emotional state.
8. Do not perform sentiment scoring or topic classification as a substitute for description.
9. Do not start Laclauian analysis. Terms such as signifier may be used descriptively, but do not label anything an empty/floating signifier, nodal point, chain of equivalence/difference, antagonism, frontier, demand, subject position, or hegemonic formation.
10. Your task is to describe the political contents in the multimodal data: the actual discourse analysis is performed in a later step. So describe, don't analyze.

### Output structure

1. **Neutral multimodal synopsis**
   - 3–8 sentences covering what the source contains and what happens across time.
   - Include only content supported by the provided modalities.

2. **Modal inventory / semiotic resources**
   - Speech/transcript
   - Written/onscreen text
   - Visual imagery
   - Gesture/posture if available
   - Spatial/compositional resources
   - Typography/graphics/symbols/emojis
   - Editing/temporal sequence
   - Sound/music only if represented in the input
   - Platform/interface/metadata context
   For unavailable modes, say "not available in supplied input".

3. **Participants, processes, circumstances**
   - Participants/entities explicitly present or named.
   - Actions/processes represented or described.
   - Relevant setting, time, place, and situational circumstances.
   - Keep explicit source naming separate from inference.
   
4. **Composition, salience, and sequence**
   - What is foregrounded/backgrounded or repeated.
   - Relative size, placement, visual hierarchy, vectors/gaze/gesture where available.
   - Changes across sampled frames and likely temporal progression.
   - Do not infer persuasion or ideology from salience alone.

5. **Salient signs / signifiers**
   - Preserve prominent and repeated words, phrases, hashtags, symbols, objects, visual motifs, sounds described in input, and gestures.
   - Prefer exact forms over normalization when useful.
   - Record language and translation uncertainty.

6. **Relational structure**
   - Observable contrasts/oppositions, pairings, repetitions, co-occurrences, sequences, labels, part-whole structures, and other sign relations.
   - These are structural observations only, not discourse-analysis conclusions.

7. **Intermodal relations**
   For each important relation between modes, describe whether they:
   - repeat/redundantly express similar content;
   - extend/complement one another;
   - elaborate/anchor/specify one another;
   - contrast or conflict;
   - remain ambiguous or disconnected.
   Note especially caption↔image, transcript↔image, OCR text↔image, and metadata↔content relations.

8. **Denotation vs cautious connotation**
   - **Denotation:** key literal observations.
   - **Possible connotations:** only conventional/culturally available associations strongly supported by the media.
   - Label connotations as possibilities, not facts. Do not convert them into ideological or political interpretation.

9. **Ambiguities, missing context, and data-quality limits**
   - Transcript uncertainty, OCR uncertainty, missing frames, unidentified persons, unclear references, unavailable audio/music, ambiguous symbols, contradictory modalities, and temporal gaps.

10. **Neutral frame / meaning-organisation description**
   - 1–3 sentences describing how the material organizes attention and presents its subject matter.
   - "Frame" here means descriptive organisation/presentation, not a political framing judgment.

11. **Named entity recognition**
   - List named entities like politicians and political parties. We are interested in political entities. 
   - The metadata may contain a human-annotated list of entities: always add these to your entity list. 
   - Entities are converted to lowercase for better matching: the entities in the metadata are lowercase.
   - The names of the entities should always be written in the same way: "marin" and "prime minister marin" should always be written "sanna marin". 

12. **Political themes**
   - List the political themes mentioned in the content.    
   - The metadata may contain a human-annotated list of political themes: always add these to your theme list. 
   - The themes should always be written in the same way: "europarliament elections" and "european elections" should always be written "ep elections".   

13. **Political sentiments**
   - Recognize political sentiments and the targets of sentiments in the content.
   - The targets of sentiments may typically be entities or themes you recognized in the previous steps.
   - Classify each sentiment as positive, negative or neutral. 
   - List the targets of positive, negative and neutral sentiments. So you need to present three simple lists of sentiment targets: positive, negative and neutral.
   - Use the same rules for writing names of entities and themes the same way and in lowercase. 

14. **TikTok/Instagram metadata**
   - Video analysis or frame analysis may have recognized TikTok/Instagram metadata like the username of the video author, video publication date or video title.
   - Create a list of this metadata, username is the most important one. 

15. **Video is problematic**
   - Note if the previous steps have noticed problems with the video.
   - If there is no transcript the video may be meaningless.
   - If the frame analysis reports there is no meaningful content in the frame the video may be useless.
   - If the video analysis reports that the video is badly cut or just garbage.
   - Report clearly if video should be DELETED or REPROCESSED. 
    
16. **Downstream-preservation block**
   - Exact salient words/phrases/hashtags.
   - Named entities explicitly present in source material.
   - Recurring visual/symbolic elements.
   - Important cross-modal contrasts or associations.

The result must be useful as evidence-preserving input to later discourse analysis while remaining methodologically distinct from that later stage.
'''
    return system_prompt


def get_llama_summary_response(system_prompt, user_prompt, *, model=None):
    """Get the configured Ollama model's response for the summary analysis."""
    selected_model = model or ollama_model()
    options = {
        "repeat_last_n": 64,
        "repeat_penalty": 1.1,
        "num_ctx": int(os.getenv("LACLAUGPT_SUMMARY_NUM_CTX", "32768")),
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "temperature": 0.0,
        "num_predict": int(os.getenv("LACLAUGPT_SUMMARY_NUM_PREDICT", "2048")),
    }
    logger.info(
        "model_call_start model=%s model_source=%s num_ctx=%s num_predict=%s",
        selected_model,
        ollama_model_source(),
        options["num_ctx"],
        options["num_predict"],
    )
    logger.debug("system_prompt=%s", system_prompt)
    logger.debug("user_prompt=%s", user_prompt)
    import ollama

    response = ollama.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=options,
    )
    llama_response = str(response["message"]["content"])
    logger.info("model_call_end model=%s response_chars=%d", selected_model, len(llama_response))
    logger.debug("llama_response=%s", llama_response)
    return llama_response


def analyze_videos(language=None):
    """Run cumulative Step 4 without dropping any upstream data."""
    started = time.monotonic()
    filename = os.getenv("LACLAUGPT_INPUT_CSV") or f"./csv/tiktok_{language}.csv"
    if not Path(filename).exists():
        logger.warning("input_missing path=%s language=%s", filename, language)
        return
    output = os.getenv("LACLAUGPT_OUTPUT_CSV") or filename
    model = ollama_model()
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)

    df = load_cumulative_csv(
        filename,
        require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")),
    )
    if max_rows > 0:
        logger.warning("demo_row_limit active=%d total=%d", max_rows, len(df))
        df = df.head(max_rows).copy()

    # Step-owned columns may be updated on an idempotent rerun. Every other
    # incoming value must remain byte-for-byte equivalent through the stage.
    before = df.drop(
        columns=[column for column in OUTPUT_COLUMNS if column in df.columns],
        errors="ignore",
    ).copy(deep=True)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    from ep24_stage_contract import STAGE_CONTRACT

    expected_prior = [
        column
        for stage in STAGE_CONTRACT
        if stage.number < 4
        for column in stage.appends
    ]
    present_prior = [column for column in expected_prior if column in df.columns]
    missing_prior = [column for column in expected_prior if column not in df.columns]

    config = StorageConfig.from_env()
    logger.info(
        "startup step=4 input=%s output=%s language=%s country=%s rows=%d columns=%d "
        "model=%s model_source=%s sqlite=%s mongo_enabled=%s mongo_db=%s mongo_collection=%s max_rows=%d",
        filename,
        output,
        language or "",
        config.country,
        len(df),
        len(df.columns),
        model,
        ollama_model_source(),
        DB_PATH,
        config.mongo_enabled,
        config.mongo_database,
        config.collection("dataframe"),
        max_rows,
    )
    logger.debug("incoming_columns=%s", list(df.columns))
    logger.info(
        "upstream_contract present=%d missing=%d",
        len(present_prior),
        len(missing_prior),
    )
    if missing_prior:
        logger.warning("upstream_contract_missing=%s", missing_prior)

    stats = {
        "processed": 0,
        "failed": 0,
        "cache_hits": 0,
        "mongo_resume_hits": 0,
        "mongo_writes": 0,
        "rag_writes": 0,
        "lock_skips": 0,
    }
    connection = _open_cache()
    storage_cm = country_storage(config.country) if config.mongo_enabled else None
    storage = storage_cm.__enter__() if storage_cm is not None else None
    redis = RedisCoordinator(config.country, 4)
    try:
        total = len(df)
        for ordinal, (index, row) in enumerate(df.iterrows(), start=1):
            source_id = _row_storage_id(row)
            author_username = ep24_value(row, "author_username")
            video_id = ep24_value(row, "video_id")
            metadata, transcript, frame_analysis, video_analysis = _evidence_from_row(row)
            retrieval_query = "\n".join(
                value for value in (
                    str(row.get("entities", "")).strip(),
                    str(row.get("themes", "")).strip(),
                    transcript[:4000],
                    frame_analysis[:2000],
                    video_analysis[:4000],
                ) if value
            )
            memory_items = []
            rag_items = []
            if storage is not None:
                try:
                    memory_items = retrieve_researcher_memory(storage, retrieval_query, limit=12)
                    rag_items = retrieve_stage_rag(
                        storage,
                        retrieval_query,
                        exclude_source_record_id=source_id,
                        limit=6,
                    )
                except Exception:
                    logger.exception("context_retrieval_failed source_id=%s", source_id)
            memory_context = _format_memory_context(memory_items)
            rag_context = _format_rag_context(rag_items)
            system_prompt = get_llama_summary_system_prompt()
            user_prompt = get_llama_summary_user_prompt(
                metadata,
                transcript,
                frame_analysis,
                video_analysis,
                memory_context,
                rag_context,
            )
            context_sha256 = _prompt_sha256(system_prompt, user_prompt, model)

            logger.info(
                "row_start ordinal=%d total=%d index=%s source_id=%s author=%s video_id=%s "
                "frame=%s ocr=%s asr=%s video=%s",
                ordinal,
                total,
                index,
                source_id,
                author_username,
                video_id,
                bool(str(row.get("frame_analysis_1", "")).strip()),
                bool(str(row.get("ocr_1", "")).strip()),
                bool(transcript),
                bool(video_analysis),
            )
            logger.debug(
                "row_context source_id=%s metadata_chars=%d transcript_chars=%d "
                "frame_chars=%d video_chars=%d memory_items=%d rag_items=%d "
                "user_prompt_chars=%d context_sha256=%s",
                source_id,
                len(metadata),
                len(transcript),
                len(frame_analysis),
                len(video_analysis),
                len(memory_items),
                len(rag_items),
                len(user_prompt),
                context_sha256,
            )

            try:
                with redis.lock(source_id) as acquired:
                    if not acquired:
                        stats["lock_skips"] += 1
                        redis.mark(source_id, "skipped_locked")
                        logger.info("redis_lock_skip source_id=%s", source_id)
                        continue

                    redis.mark(source_id, "running")
                    durable = (
                        _mongo_resume_summary(
                            storage, source_id, model=model, context_sha256=context_sha256
                        )
                        if storage is not None
                        else None
                    )
                    cached = durable or _cache_lookup(connection, source_id, model, context_sha256)
                    if durable is not None:
                        stats["mongo_resume_hits"] += 1
                        logger.info(
                            "mongo_resume_hit source_id=%s response_chars=%d",
                            source_id,
                            len(durable),
                        )

                    if cached is not None:
                        summary_analysis = cached
                        stats["cache_hits"] += 1
                        logger.info(
                            "cache_hit source_id=%s response_chars=%d",
                            source_id,
                            len(cached),
                        )
                    else:
                        summary_analysis = get_llama_summary_response(
                            system_prompt,
                            user_prompt,
                            model=model,
                        )
                        if not summary_analysis.strip():
                            raise ValueError("summary model returned an empty response")
                        _cache_store(
                            connection,
                            source_id=source_id,
                            model=model,
                            context_sha256=context_sha256,
                            summary_analysis=summary_analysis,
                        )
                        logger.debug("cache_store source_id=%s", source_id)

                    df.at[index, "metadata"] = metadata
                    df.at[index, "summary_analysis"] = summary_analysis
                    df.at[index, "summary_summary_md"] = summary_analysis
                    stats["processed"] += 1

                    # CSV remains the cumulative interchange/checkpoint artifact.
                    write_cumulative_csv(before, df, output)
                    logger.info(
                        "local_checkpoint source_id=%s output=%s fields=%s",
                        source_id,
                        output,
                        OUTPUT_COLUMNS,
                    )

                    if storage is not None:
                        try:
                            count = _persist_mongo_row(
                                storage,
                                df.loc[index],
                                source_id=source_id,
                                model=model,
                                context_sha256=context_sha256,
                            )
                            stats["mongo_writes"] += count
                            rag_count = upsert_stage_rag(
                                storage,
                                df.loc[[index]].assign(_storage_id=source_id),
                                stage="summary",
                            )
                            stats["rag_writes"] += rag_count
                            logger.info(
                                "mongo_status source_id=%s dataframe_writes=%d rag_writes=%d",
                                source_id,
                                count,
                                rag_count,
                            )
                        except Exception:
                            logger.exception(
                                "mongo_persist_failed source_id=%s local_checkpoint_is_safe=true",
                                source_id,
                            )
                            # MongoDB is the durable shared source of truth. The
                            # already-written CSV/cache make retry safe, but the
                            # record must not be reported completed until Mongo
                            # persistence succeeds.
                            raise
                    redis.mark(source_id, "completed")
            except Exception as exc:
                stats["failed"] += 1
                redis.mark(source_id, "failed")
                logger.exception(
                    "row_failed index=%s source_id=%s video_id=%s error=%s",
                    index,
                    source_id,
                    video_id,
                    exc,
                )

        # Ensure even an all-failure/empty run materializes the stage-owned columns.
        write_cumulative_csv(before, df, output)
    finally:
        connection.close()
        logger.debug("sqlite_closed path=%s", DB_PATH)
        if storage_cm is not None:
            storage_cm.__exit__(None, None, None)
            logger.debug("mongo_closed country=%s", config.country)

    logger.info(
        "complete step=4 processed=%d failed=%d cache_hits=%d mongo_resume_hits=%d "
        "mongo_writes=%d rag_writes=%d lock_skips=%d output=%s elapsed_seconds=%.3f",
        stats["processed"],
        stats["failed"],
        stats["cache_hits"],
        stats["mongo_resume_hits"],
        stats["mongo_writes"],
        stats["rag_writes"],
        stats["lock_skips"],
        output,
        time.monotonic() - started,
    )


# Loop through each EP2024 TikTok language and analyze videos
# All EP2024 TikTok languages for this stage (module level: the documented
# stage contract reads it without importing or executing the stage).
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']


if __name__ == "__main__":
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        analyze_videos(None)
    else:
        for language in languages:
            analyze_videos(language)
