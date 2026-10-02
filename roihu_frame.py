import pandas as pd
import logging
import os
import cv2
import ollama
import base64
import ast
import sqlite3
from logging.handlers import RotatingFileHandler
from ep24_pipeline import load_cumulative_csv, metadata_context
from ep24_schema import value as ep24_value
from ep24_video import VIDEO_INITIAL_SKIP_SECONDS
logger = logging.getLogger(__name__)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./database', exist_ok=True)
logging.basicConfig(handlers=[RotatingFileHandler('./logs/frame.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Convert this to use the remote MongoDB database?
# You can use Pandas dataframe CSV / Sqlite as backup of data 
# Use sqlite3 database to store TikTok video frame analysis results
conn = sqlite3.connect('./database/frame.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tiktok_videos
                (author_username text,
                video_id text,
                frame_analysis_1 text,
                frame_analysis_2 text,
                frame_analysis_3 text,
                frame_analysis_4 text,
                frame_analysis_5 text,
                frame_analysis_6 text)''')
conn.commit()

# Get the analysis from Ollama
def get_analysis(frame_file, row_context=''):
    """Analyze a single frame from a TikTok video using the Llama model."""
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
        response = ollama.chat(model=os.getenv('LACLAUGPT_MULTIMODAL_MODEL', 'gemma4:12b'), 
                               messages=[
                                    {'role': 'system', 'content': system_prompt}, 
                                    {'role': 'user', 'content': user_prompt, 'images': images},
                                    ], options=options)
        frame_message = response['message']
        frame_analysis = frame_message['content']
        logger.debug(f'Frame description: {frame_analysis}')
    except Exception as e:
        logger.error(f'Error processing image: {e}')
    return frame_analysis

def parse_frame_files(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(',') if item.strip()]


def analyze_videos(language=None):
    """Analyze TikTok videos for a specific language."""
    # Use correct filename / SQLITE / Remote Mongo for incoming data
    # Loop through all videos of each country in specified order.
    filename = os.getenv('LACLAUGPT_INPUT_CSV') or f'./csv/tiktok_{language}.csv' 
    df = load_cumulative_csv(filename, require_canonical=bool(os.getenv('LACLAUGPT_INPUT_CSV')))
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        df = df.head(max_rows).copy()
        logger.info("Demo row limit active: processing first %s rows", max_rows)
    # Historical per-language mode keeps its documented row-drop semantics.
    # Canonical EP24 reprocessing is additive and keeps every source row.
    if not os.getenv('LACLAUGPT_INPUT_CSV'):
        df = df.dropna(subset=['whisperResult'])
        df = df.dropna(subset=['frame_files'])
    for column in (
        'frame_analysis_1',
        'frame_analysis_2',
        'frame_analysis_3',
        'frame_analysis_4',
        'frame_analysis_5',
        'frame_analysis_6',
        'frame_analysis_timestamp_seconds',
        'frame_analysis_status',
    ):
        if column not in df.columns:
            df[column] = ''
    if language and 'language' in df.columns:
        df = df[df['language'] == language].copy()
    for (index, row) in df.iterrows():
        author_username = ep24_value(row, 'author_username')
        video_id = ep24_value(row, 'video_id')
        # Check if exists in database
        c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
        if c.fetchone():
            logger.debug(f'Video already processed: {author_username} - {video_id}')
            # Get All from the database
            c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
            # Get all from the database
            row = c.fetchone()
            df.at[index, 'frame_analysis_1'] = str(row[2] or '')
            df.at[index, 'frame_analysis_timestamp_seconds'] = str(VIDEO_INITIAL_SKIP_SECONDS)
            df.at[index, 'frame_analysis_status'] = 'cached'
            # Historical cache rows can contain six frame analyses. Current
            # production semantics expose only the t=1.0s contextual frame.
            for old_index in range(2, 7):
                df.at[index, f'frame_analysis_{old_index}'] = ''
        else:
            try:
                frame_files = parse_frame_files(row['frame_files'])
                if not frame_files:
                    raise ValueError('Step 2 requires the Step 1 keyframe at original t=1.0s')

                # Production contract: analyze one and only one still image.
                # Step 1 guarantees frame_files[0] is extracted at original t=1.0s,
                # immediately after the known feed-scroll artifact. Temporal coverage
                # belongs to Step 3 native-video analysis, not repeated still sampling.
                frame_file = frame_files[0]
                frame_response = str(get_analysis(frame_file, metadata_context(row)))
                seconds = str(VIDEO_INITIAL_SKIP_SECONDS)
                frame_analysis_1 = f'''### **Frame 1 at original t={seconds} seconds**:
{frame_response}
'''
                frame_analysis_2 = ""
                frame_analysis_3 = ""
                frame_analysis_4 = ""
                frame_analysis_5 = ""
                frame_analysis_6 = ""

                logger.debug('Single-frame analysis at original t=%ss: %s', seconds, frame_analysis_1)
                df.at[index, 'frame_analysis_1'] = frame_analysis_1
                df.at[index, 'frame_analysis_timestamp_seconds'] = seconds
                df.at[index, 'frame_analysis_status'] = 'ok'
                for old_index in range(2, 7):
                    df.at[index, f'frame_analysis_{old_index}'] = ''
                # Insert to database if not exists
                c.execute("INSERT INTO tiktok_videos (author_username, video_id, frame_analysis_1, frame_analysis_2, frame_analysis_3, frame_analysis_4, frame_analysis_5, frame_analysis_6) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",(str(author_username), str(video_id), str(frame_analysis_1), str(frame_analysis_2), str(frame_analysis_3), str(frame_analysis_4), str(frame_analysis_5), str(frame_analysis_6)))
                conn.commit()
            except Exception as e:
                df.at[index, 'frame_analysis_status'] = 'error'
                logger.exception('Error processing single t=1.0s frame: %s', e)
    output = os.getenv('LACLAUGPT_OUTPUT_CSV') or filename
    df.to_csv(output, index=False)


# Loop through all EP2024 TikTok languages and analyze videos
# All EP2024 TikTok languages for this stage (module level: the documented
# stage contract reads it without importing or executing the stage).
# Use the country list instead of this.
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


