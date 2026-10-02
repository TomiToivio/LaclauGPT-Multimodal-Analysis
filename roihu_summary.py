import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler

import ollama
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import load_cumulative_csv, metadata_context
from ep24_schema import value as ep24_value
logger = logging.getLogger(__name__)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./database', exist_ok=True)
logging.basicConfig(handlers=[RotatingFileHandler('./logs/summary.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Sqlite3 database connection
conn = sqlite3.connect('./database/summary.db')
c = conn.cursor()
# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS tiktok_videos
                (author_username text, 
                video_id text,
                summary_analysis text)''')
conn.commit()

# Note: Only one Frame analysis from now on.
# Add video analysis
# Transcript is good to be here
# Put the rest of the dataframe columns in metadata. I mean every field the pipeline has produced so far. 
def get_llama_summary_user_prompt(metadata, transcript, frame_analysis, video_analysis):
    """Construct the user prompt for social-semiotic multimodal pre-analysis."""
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
      
15. **Downstream-preservation block**
   - Exact salient words/phrases/hashtags.
   - Named entities explicitly present in source material.
   - Recurring visual/symbolic elements.
   - Important cross-modal contrasts or associations.

The result must be useful as evidence-preserving input to later discourse analysis while remaining methodologically distinct from that later stage.
'''
    return system_prompt


def get_llama_summary_response(system_prompt, user_prompt):
    """Get the Llama model's response for the summary analysis."""
    options = {"repeat_last_n": 64,
               "repeat_penalty": 1.1,
               "num_ctx": 10240,
               "top_p": 0.9,
               "top_k": 40,
               "min_p": 0.0,
               "temperature": 0.0,
               "num_predict": 2048}
    logger.info("model=%s model_source=%s", ollama_model(), ollama_model_source())
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug(f"User prompt: {user_prompt}")
    response = ollama.chat(model=ollama_model(), messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
    ], options=options)
    llama_response = response['message']['content']
    logger.debug(f"LLAMA response: {llama_response}")
    return llama_response


def analyze_videos(language=None):
    """Fuse all upstream evidence without dropping canonical EP24 metadata."""
    filename = os.getenv('LACLAUGPT_INPUT_CSV') or f'./csv/tiktok_{language}.csv'
    df = load_cumulative_csv(filename, require_canonical=bool(os.getenv('LACLAUGPT_INPUT_CSV')))
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        df = df.head(max_rows).copy()
        logger.info("Demo row limit active: processing first %s rows", max_rows)

    # Preserve documented historical behavior without dropping canonical rows.
    if not os.getenv('LACLAUGPT_INPUT_CSV') and 'whisperResult' in df.columns:
        df = df.dropna(subset=['whisperResult'])

    if 'summary_analysis' not in df.columns:
        df['summary_analysis'] = ''
    if 'summary_summary_md' not in df.columns:
        df['summary_summary_md'] = ''

    for index, row in df.iterrows():
        author_username = ep24_value(row, 'author_username')
        video_id = ep24_value(row, 'video_id')
        logger.debug('Analyzing video %s', video_id)

        metadata = metadata_context(row)
        transcript = (
            str(row.get('asr_translated', '')).strip()
            or str(row.get('asr_transcript', '')).strip()
            # Temporary read-only compatibility for frozen legacy artifacts.
            or str(row.get('whisper_translated', '')).strip()
            or str(row.get('whisper_transcript', '')).strip()
            or str(row.get('whisperResult', '')).strip()
        )

        frame_parts = []
        frame_text = str(row.get('frame_analysis_1', '')).strip()
        ocr_text = str(row.get('ocr_1', '')).strip()
        if frame_text:
            frame_parts.append(frame_text)
        if ocr_text:
            frame_parts.append(f"### OCR frame at original t=1.0s\n{ocr_text}")
        video_text = str(row.get('vllm_video_analysis', '')).strip()
        if video_text:
            frame_parts.append("### Whole-video analysis\n" + video_text)
        frame_analysis = "\n\n".join(frame_parts)

        c.execute(
            "SELECT summary_analysis FROM tiktok_videos WHERE author_username = ? AND video_id = ?",
            (str(author_username), str(video_id)),
        )
        cached = c.fetchone()
        if cached:
            summary_analysis = str(cached[0] or '')
        else:
            try:
                user_prompt = get_llama_summary_user_prompt(metadata, transcript, frame_analysis)
                system_prompt = get_llama_summary_system_prompt()
                summary_analysis = get_llama_summary_response(system_prompt, user_prompt)
                c.execute(
                    "INSERT INTO tiktok_videos (author_username, video_id, summary_analysis) VALUES (?, ?, ?)",
                    (str(author_username), str(video_id), str(summary_analysis)),
                )
                conn.commit()
            except Exception as exc:
                logger.exception('Error processing video %s: %s', video_id, exc)
                summary_analysis = ''

        df.at[index, 'metadata'] = metadata
        df.at[index, 'summary_analysis'] = summary_analysis
        df.at[index, 'summary_summary_md'] = summary_analysis

    output = os.getenv('LACLAUGPT_OUTPUT_CSV') or filename
    df.to_csv(output, index=False)

# Loop through each EP2024 TikTok language and analyze videos
# All EP2024 TikTok languages for this stage (module level: the documented
# stage contract reads it without importing or executing the stage).
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg', 'en']


if __name__ == '__main__':
    try:
        if os.getenv('LACLAUGPT_INPUT_CSV'):
            analyze_videos(None)
        else:
            for language in languages:
                analyze_videos(language)
    finally:
        c.close()
        conn.close()


