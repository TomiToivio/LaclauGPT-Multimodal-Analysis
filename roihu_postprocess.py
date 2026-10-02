import json
import logging
import os

import ollama
import pandas as pd
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel
from ep24_pipeline import ensure_columns, load_cumulative_csv, metadata_context
from ep24_schema import stable_source_id, value as ep24_value

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


class Sentiment(BaseModel):
    topics: list[str]
    themes: list[str]
    positive: list[str]
    neutral: list[str]
    negative: list[str]


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
    options = {
        'repeat_last_n': 64,
        'repeat_penalty': 1.1,
        'num_ctx': 4096,
        'top_p': 0.9,
        'top_k': 40,
        'min_p': 0.0,
        'temperature': 0.0,
        'num_predict': 2048,
    }
    try:
        response = ollama.chat(
            model=os.getenv('LACLAUGPT_MULTIMODAL_MODEL', 'gemma4:12b'),
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options=options,
            format=Sentiment.model_json_schema(),
        )
        llama_response = response['message']['content']
        logger.debug(llama_response)
        return Sentiment.model_validate_json(llama_response)
    except Exception as exc:
        logger.error('Error getting structured response: %s', exc)
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
    return [item.strip() for item in text.split(",") if item.strip()]


def analyze_responses(language=None):
    filename = os.getenv('LACLAUGPT_INPUT_CSV') or source_filename(language)
    if filename is None:
        logger.warning('No input file found for language %s; skipping', language)
        return

    df = load_cumulative_csv(filename, require_canonical=bool(os.getenv('LACLAUGPT_INPUT_CSV')))
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        df = df.head(max_rows).copy()
        logger.info("Demo row limit active: processing first %s rows", max_rows)
    if 'summary_analysis' not in df.columns:
        logger.error('Missing summary_analysis in %s; skipping', filename)
        return

    df = ensure_video_filename(df)
    ensure_columns(df, ('entities', 'themes', 'positive', 'neutral', 'negative', 'postprocess_summary_md'))

    system_prompt = get_system_prompt()

    for index, row in df.iterrows():
        logger.info('Processing row %s of %s', index, filename)
        summary_analysis = row.get('summary_analysis')
        if pd.isna(summary_analysis) or not str(summary_analysis).strip():
            logger.warning('Row %s has no summary_analysis; skipping', index)
            continue

        prompt = metadata_context(row) + '\n\nSUMMARY EVIDENCE:\n' + str(summary_analysis)
        response = get_response(prompt, system_prompt)
        if response is None:
            continue

        values = {
            'entities': response.entities,
            'themes': response.themes,
            'positive': response.positive,
            'neutral': response.neutral,
            'negative': response.negative,
        }
        for column, items in values.items():
            # Preserve upstream cumulative values (especially bootstrap entities)
            # and append model discoveries without exact duplicates.
            existing = _existing_list(row.get(column, ""))
            combined = [*existing, *(str(item) for item in items if str(item))]
            unique_items = list(dict.fromkeys(item for item in combined if item))
            df.at[index, column] = ', '.join(unique_items)
        df.at[index, 'postprocess_summary_md'] = (
            '**Entities:** ' + df.at[index, 'entities'] + '\n\n'
            + '**Themes/topics:** ' + df.at[index, 'themes'] + '\n\n'
            + '**Sentiment targets:** positive=' + df.at[index, 'positive']
            + '; neutral=' + df.at[index, 'neutral']
            + '; negative=' + df.at[index, 'negative']
        )

    output = os.getenv('LACLAUGPT_OUTPUT_CSV') or filename
    df.to_csv(output, index=False)

    # roihu_populism.py is a legacy consumer of ep24_<language>.csv. Emit that
    # compatibility artifact so the documented sequence works end-to-end.
    if not os.getenv('LACLAUGPT_INPUT_CSV') and language:
        legacy_output = f'ep24_{language}.csv'
        if filename != legacy_output:
            df.to_csv(legacy_output, index=False)


languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']


if __name__ == '__main__':
    if os.getenv('LACLAUGPT_INPUT_CSV'):
        analyze_responses(None)
    else:
        for language in languages:
            analyze_responses(language)
