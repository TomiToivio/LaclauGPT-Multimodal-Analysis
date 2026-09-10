import logging
import os

import ollama
import pandas as pd
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel

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
    entities: list[str]
    positive: list[str]
    neutral: list[str]
    negative: list[str]


def get_system_prompt():
    return '''### **System Prompt**

**Role**:
- You are presented a previously generated analysis of a political video.
- Extract entities, sentiments and topics from the analysis.
- Provide simple lists of names or sentiment targets.
- If a category has no values, return an empty list.

**Tasks**:
1. Extract political topics, merging obvious duplicates or synonyms.
2. Extract political entities, merging obvious duplicates or synonyms.
3. Extract sentiment targets and classify each as positive, neutral, or negative.

**Formatting Rules**:
- Respond only with a valid JSON object.
- Use exactly these keys: `topics`, `entities`, `positive`, `neutral`, `negative`.
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
            model='gemma3:27b',
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
    """Create the cache key required by puhti_populism.py when it is absent."""
    if 'video_filename' in df.columns:
        return df

    if {'authorUniqueId', 'videoId'}.issubset(df.columns):
        df['video_filename'] = (
            df['authorUniqueId'].astype(str)
            + '/'
            + df['videoId'].astype(str)
        )
    elif 'videoId' in df.columns:
        df['video_filename'] = df['videoId'].astype(str)
    else:
        # Last-resort stable key for legacy datasets without TikTok identifiers.
        df['video_filename'] = df.index.astype(str)
        logger.warning(
            'Input has no videoId/authorUniqueId; using row index as video_filename'
        )
    return df


def analyze_responses(language):
    filename = source_filename(language)
    if filename is None:
        logger.warning('No input file found for language %s; skipping', language)
        return

    df = pd.read_csv(filename)
    if 'summary_analysis' not in df.columns:
        logger.error('Missing summary_analysis in %s; skipping', filename)
        return

    df = ensure_video_filename(df)
    for column in ('entities', 'topics', 'positive', 'neutral', 'negative'):
        df[column] = ''

    system_prompt = get_system_prompt()

    for index, row in df.iterrows():
        logger.info('Processing row %s of %s', index, filename)
        summary_analysis = row.get('summary_analysis')
        if pd.isna(summary_analysis) or not str(summary_analysis).strip():
            logger.warning('Row %s has no summary_analysis; skipping', index)
            continue

        response = get_response(str(summary_analysis), system_prompt)
        if response is None:
            continue

        values = {
            'entities': response.entities,
            'topics': response.topics,
            'positive': response.positive,
            'neutral': response.neutral,
            'negative': response.negative,
        }
        for column, items in values.items():
            # Preserve order while removing exact duplicates.
            unique_items = list(dict.fromkeys(str(item) for item in items if str(item)))
            df.at[index, column] = ', '.join(unique_items)

    # Keep the canonical pipeline file up to date.
    df.to_csv(filename, index=False)

    # puhti_populism.py is a legacy consumer of ep24_<language>.csv. Emit that
    # compatibility artifact so the documented sequence works end-to-end.
    legacy_output = f'ep24_{language}.csv'
    if filename != legacy_output:
        df.to_csv(legacy_output, index=False)


languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']
for language in languages:
    analyze_responses(language)
