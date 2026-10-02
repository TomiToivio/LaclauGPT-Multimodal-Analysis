import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from logging.handlers import RotatingFileHandler
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import ensure_columns, load_cumulative_csv, metadata_context, write_cumulative_csv
from ep24_schema import stable_source_id, value as ep24_value
from roihu_storage import MongoStorage, StorageConfig

os.makedirs('./logs', exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        RotatingFileHandler(
            './logs/postprocess.log',
            encoding='utf-8',
            maxBytes=1000000,
            backupCount=5,
        )
    ],
    level=logging.DEBUG,
)


OUTPUT_COLUMNS = (
    "video_filename",
    "postprocess_entities",
    "postprocess_themes",
    "positive",
    "neutral",
    "negative",
    "postprocess_summary_md",
)
ENRICHMENT_COLUMNS = (
    "ep24_entity_resolution_json",
    "ep24_entity_ids",
    "ep24_entity_canonical_names",
    "ep24_entity_unresolved_json",
    "ep24_codebook_fingerprint",
    "ep24_codebook_context_json",
    "ep24_memory_entity_ids",
    "ep24_memory_topic_ids",
    "ep24_memory_theme_ids",
    "ep24_theme_resolution_json",
    "ep24_theme_canonical_names",
    "ep24_memory_sentiment_target_ids_json",
    "ep24_memory_unresolved_json",
    "ep24_seed_entities_json",
    "ep24_seed_themes_json",
    "ep24_sentiment_targets_json",
    "ep24_human_seed_context",
)
PROMPT_EXCLUDE_COLUMNS = {
    "summary_analysis",
    "metadata",
    "vllm_video_raw_output",
    "vllm_video_prompt",
    "vllm_structured_output",
}


def _result_model():
    """Build the structured output schema lazily so CLI/import stays lightweight."""
    from pydantic import BaseModel

    class PostprocessResult(BaseModel):
        entities: list[str]
        themes: list[str]
        positive: list[str]
        neutral: list[str]
        negative: list[str]

    return PostprocessResult


def postprocess_num_ctx() -> int:
    return int(os.getenv("LACLAUGPT_POSTPROCESS_NUM_CTX", "32768"))


def postprocess_num_predict() -> int:
    return int(os.getenv("LACLAUGPT_POSTPROCESS_NUM_PREDICT", "2048"))


def _prompt_context(row: pd.Series) -> str:
    """Cumulative context without duplicating the summary or giant raw prompt blobs."""
    reduced = row.drop(labels=[c for c in PROMPT_EXCLUDE_COLUMNS if c in row.index])
    return metadata_context(reduced, include_model_fields=True)


def build_postprocess_prompt(row: pd.Series) -> str:
    summary = str(row.get("summary_analysis", "") or "").strip()
    return _prompt_context(row) + "\n\nSUMMARY EVIDENCE:\n" + summary


def _country_code(row: pd.Series, language: str | None = None) -> str:
    from roihu_codebooks import COUNTRY_PROFILES

    raw = str(row.get("country", "") or os.getenv("LACLAUGPT_COUNTRY", "")).strip()
    if raw.upper() in COUNTRY_PROFILES:
        return raw.upper()
    by_name = {
        str(meta["country"]).casefold(): code
        for code, meta in COUNTRY_PROFILES.items()
    }
    if raw.casefold() in by_name:
        return by_name[raw.casefold()]
    if language:
        matches = [
            code for code, meta in COUNTRY_PROFILES.items()
            if language in meta.get("languages", []) and language != "en"
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"cannot resolve EP24 country code from country={raw!r} language={language!r}")


def _patch_mongo_dataframe(frame: pd.DataFrame, *, fields: tuple[str, ...], stage: str) -> str:
    config = StorageConfig.from_env()
    if not config.mongo_enabled:
        logger.info("mongo persistence disabled by LACLAUGPT_MONGO_ENABLED")
        return "mongo_disabled"
    storage = MongoStorage(config)
    try:
        documents = []
        for _, row in frame.iterrows():
            source_id = stable_source_id(row)
            doc = {"_storage_id": source_id}
            for field in fields:
                if field in frame.columns:
                    doc[field] = row.get(field, "")
            doc[f"_provenance.{stage}"] = {
                "pipeline_stage": stage,
                "model": ollama_model(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            documents.append(doc)
        count = storage.patch_documents("dataframe", documents)
        logger.info(
            "mongo_patch stage=%s rows=%d collection=%s",
            stage,
            count,
            storage.collection_name("dataframe"),
        )
        return f"mongo_ok:{count}"
    finally:
        storage.close()


def get_system_prompt():
    return '''### **System Prompt**

**Role**:
- You are presented a previously generated analysis of a political video.
- Extract entities, sentiments and themes from the analysis.
- Provide simple lists of names or sentiment targets.
- If a category has no values, return an empty list.

**Tasks**:
1. Extract political themes, merging obvious duplicates or synonyms.
2. Extract political entities, merging obvious duplicates or synonyms.
3. Extract sentiment targets and classify each as positive, neutral, or negative.

**Formatting Rules**:
- Respond only with a valid JSON object.
- Use exactly these keys: `themes`, `entities`, `positive`, `neutral`, `negative`.
- Every value must be a JSON array of strings.
- Do not include introductions, markdown fences, comments, or extra fields.
'''


def get_response(user_prompt, system_prompt):
    ResultModel = _result_model()
    num_ctx = postprocess_num_ctx()
    options = {
        "repeat_last_n": 64,
        "repeat_penalty": 1.1,
        "num_ctx": num_ctx,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "temperature": 0.0,
        "num_predict": postprocess_num_predict(),
    }
    approx_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
    logger.info(
        "model_call_start model=%s model_source=%s num_ctx=%d num_predict=%d "
        "prompt_chars=%d approx_tokens=%d",
        ollama_model(),
        ollama_model_source(),
        num_ctx,
        options["num_predict"],
        len(system_prompt) + len(user_prompt),
        approx_tokens,
    )
    if approx_tokens > int(num_ctx * 0.80):
        logger.warning(
            "context_budget_high approx_tokens=%d num_ctx=%d utilization=%.2f",
            approx_tokens,
            num_ctx,
            approx_tokens / num_ctx,
        )
    try:
        import ollama

        response = ollama.chat(
            model=ollama_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
            format=ResultModel.model_json_schema(),
        )
        llama_response = str(response["message"]["content"])
        logger.debug("structured_response=%s", llama_response)
        return ResultModel.model_validate_json(llama_response)
    except Exception as exc:
        logger.exception("structured postprocess failed: %s", exc)
        return None


def source_filename(language):
    """Find the previous pipeline stage while retaining legacy compatibility."""
    pipeline_file = f'./csv/tiktok_{language}.csv'
    legacy_file = f'ep24_{language}.csv'

    if os.path.exists(pipeline_file):
        return pipeline_file
    if os.path.exists(legacy_file):
        logger.warning('Using legacy input path %s', legacy_file)
        return legacy_file

    return None


def ensure_video_filename(df):
    """Compatibility cache key derived from canonical EP24 identity."""
    if 'video_filename' in df.columns:
        return df
    df['video_filename'] = [
        ep24_value(row, 'allas_filename') or stable_source_id(row) or str(index)
        for index, row in df.iterrows()
    ]
    return df


def _existing_list(value):
    """Parse cumulative list-like fields without discarding researcher seeds."""
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    for separator in ("\n", "|"):
        text = text.replace(separator, ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def analyze_responses(language=None):
    started = time.monotonic()
    filename = os.getenv("LACLAUGPT_INPUT_CSV") or source_filename(language)
    if filename is None or not Path(filename).exists():
        logger.warning("input_missing language=%s path=%s", language, filename)
        return

    output = os.getenv("LACLAUGPT_OUTPUT_CSV") or filename
    df = load_cumulative_csv(
        filename,
        require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")),
    )
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        df = df.head(max_rows).copy()
        logger.info("demo_row_limit active=%d", max_rows)
    if "summary_analysis" not in df.columns:
        raise ValueError(f"Step 5 requires Step 4 field summary_analysis in {filename}")

    # These fields are legitimate Step-5 outputs/updates. Everything else is
    # immutable upstream state and is checked by write_cumulative_csv().
    before = df.drop(
        columns=[
            column for column in (*OUTPUT_COLUMNS, *ENRICHMENT_COLUMNS)
            if column in df.columns
        ],
        errors="ignore",
    ).copy(deep=True)
    df = ensure_video_filename(df)
    ensure_columns(
        df,
        (
            "entities",
            "themes",
            "postprocess_entities",
            "postprocess_themes",
            "positive",
            "neutral",
            "negative",
            "postprocess_summary_md",
        ),
    )

    from ep24_stage_contract import STAGE_CONTRACT

    expected_prior = [
        column
        for stage in STAGE_CONTRACT
        if stage.number <= 4
        for column in stage.appends
    ]
    missing_prior = [column for column in expected_prior if column not in df.columns]
    config = StorageConfig.from_env()
    logger.info(
        "startup step=5 input=%s output=%s rows=%d columns=%d language=%s model=%s "
        "num_ctx=%d num_predict=%d mongo_enabled=%s mongo_db=%s mongo_collection=%s",
        filename,
        output,
        len(df),
        len(df.columns),
        language or "",
        ollama_model(),
        postprocess_num_ctx(),
        postprocess_num_predict(),
        config.mongo_enabled,
        config.mongo_database,
        config.collection("dataframe"),
    )
    logger.debug("incoming_columns=%s", list(df.columns))
    if missing_prior:
        logger.warning("upstream_contract_missing=%s", missing_prior)

    system_prompt = get_system_prompt()
    stats = {"processed": 0, "skipped": 0, "failed": 0, "parse_failures": 0}
    total = len(df)

    for ordinal, (index, row) in enumerate(df.iterrows(), start=1):
        source_id = stable_source_id(row)
        summary_analysis = str(row.get("summary_analysis", "") or "").strip()
        if not summary_analysis:
            stats["skipped"] += 1
            logger.warning(
                "row_skip ordinal=%d total=%d index=%s source_id=%s reason=missing_summary",
                ordinal,
                total,
                index,
                source_id,
            )
            continue

        prompt = build_postprocess_prompt(row)
        logger.info(
            "row_start ordinal=%d total=%d index=%s source_id=%s summary_chars=%d "
            "prompt_chars=%d existing_entities=%d existing_themes=%d",
            ordinal,
            total,
            index,
            source_id,
            len(summary_analysis),
            len(prompt),
            len(_existing_list(row.get("entities", ""))),
            len(_existing_list(row.get("themes", ""))),
        )
        response = get_response(prompt, system_prompt)
        if response is None:
            stats["parse_failures"] += 1
            continue

        # Issue #21: entities/themes are authoritative human annotations.
        # Machine extraction is additive and must never overwrite or extend them.
        machine_values = {
            "postprocess_entities": response.entities,
            "postprocess_themes": response.themes,
            "positive": response.positive,
            "neutral": response.neutral,
            "negative": response.negative,
        }
        for column, items in machine_values.items():
            existing = _existing_list(row.get(column, ""))
            combined = [*existing, *(str(item).strip() for item in items if str(item).strip())]
            df.at[index, column] = json.dumps(list(dict.fromkeys(combined)), ensure_ascii=False)

        df.at[index, "postprocess_summary_md"] = (
            "**Machine entities:** " + str(df.at[index, "postprocess_entities"]) + "\n\n"
            + "**Machine themes:** " + str(df.at[index, "postprocess_themes"]) + "\n\n"
            + "**Human entities:** " + str(df.at[index, "entities"]) + "\n\n"
            + "**Human themes:** " + str(df.at[index, "themes"]) + "\n\n"
            + "**Sentiment targets:** positive=" + str(df.at[index, "positive"])
            + "; neutral=" + str(df.at[index, "neutral"])
            + "; negative=" + str(df.at[index, "negative"])
        )
        stats["processed"] += 1
        logger.info(
            "row_extracted source_id=%s entities=%d themes=%d positive=%d neutral=%d negative=%d",
            source_id,
            len(response.entities),
            len(response.themes),
            len(response.positive),
            len(response.neutral),
            len(response.negative),
        )

    # First durability boundary: raw extraction plus every upstream field.
    write_cumulative_csv(before, df, output)
    logger.info("local_checkpoint output=%s rows=%d", output, len(df))

    try:
        _patch_mongo_dataframe(df, fields=OUTPUT_COLUMNS, stage="step5_postprocess")
    except Exception:
        logger.exception("mongo_postprocess_failed local_checkpoint_is_safe=true")

    # Canonical cleanup belongs immediately after Step 5, before Step 6.
    # Reuse the existing codebook/memory resolver rather than inventing another
    # identity system. The raw entity/theme/sentiment strings remain untouched.
    try:
        from roihu_enrich import enrich_file
        from roihu_memory import EP24Memory

        country = _country_code(df.iloc[0] if len(df) else pd.Series(dtype=str), language)
        private_root = Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "."))
        memory = EP24Memory(
            os.getenv("LACLAUGPT_MEMORY_DB", "./database/ep24_memory.sqlite3")
        )
        report = enrich_file(
            Path(output),
            country=country,
            language=language or str(df.iloc[0].get("language", "") or ""),
            private_root=private_root,
            memory=memory,
        )
        logger.info(
            "normalization_complete country=%s rows=%s entity_report=%s",
            country,
            report.get("rows"),
            report.get("entity_resolution"),
        )
        normalized = load_cumulative_csv(output, require_canonical=False)
        # Verify enrichment was additive relative to the Step-5 checkpoint.
        for column in df.columns:
            if normalized[column].astype(str).tolist() != df[column].astype(str).tolist():
                raise AssertionError(f"normalization mutated raw Step-5 field: {column}")
        df = normalized
        try:
            normalized_fields = tuple(
                column for column in ENRICHMENT_COLUMNS if column in df.columns
            )
            _patch_mongo_dataframe(
                df,
                fields=normalized_fields,
                stage="step5_normalization",
            )
        except Exception:
            logger.exception("mongo_normalization_failed local_checkpoint_is_safe=true")
    except Exception:
        stats["failed"] += 1
        logger.exception(
            "postprocess_normalization_failed output=%s; raw Step-5 checkpoint remains safe",
            output,
        )

    # Keep legacy compatibility only outside the explicit cumulative pipeline.
    if not os.getenv("LACLAUGPT_INPUT_CSV") and language:
        legacy_output = f"ep24_{language}.csv"
        if str(output) != legacy_output:
            df.to_csv(legacy_output, index=False)

    logger.info(
        "complete step=5 processed=%d skipped=%d failed=%d parse_failures=%d "
        "output=%s elapsed_seconds=%.3f",
        stats["processed"],
        stats["skipped"],
        stats["failed"],
        stats["parse_failures"],
        output,
        time.monotonic() - started,
    )



languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']


if __name__ == '__main__':
    if os.getenv('LACLAUGPT_INPUT_CSV'):
        analyze_responses(None)
    else:
        for language in languages:
            analyze_responses(language)
