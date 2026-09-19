import pandas as pd
import logging
import os
import cv2
import ollama
import base64
import sqlite3
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[RotatingFileHandler('./logs/frame.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
def get_analysis(frame_file):
    """Analyze a single frame from a TikTok video using the Llama model."""
    # Social-semiotic first-pass prompt. Keep this stage descriptive and pre-discursive.
    system_prompt = f'''### System Prompt

You are performing a **multimodal social-semiotic pre-analysis** of a single sampled frame from incoming social-media or web video.

This is an upstream descriptive stage. **Do not perform political, ideological, partisan, populism, sentiment, discourse, or Laclauian analysis.** Do not classify empty/floating signifiers, nodal points, chains of equivalence, antagonisms, hegemony, political camps, motives, or persuasive effectiveness. Those tasks belong to later analytical stages.

Use a light social-semiotic methodology inspired by Halliday/SFL, Kress & van Leeuwen, multimodal social semiotics, and structuralist attention to signs and relations. Separate observation from interpretation and mark uncertainty explicitly.

### Input
- One sampled video frame. It may contain people, objects, environments, captions, subtitles, memes, screenshots, platform UI, graphics, diagrams, logos, symbols, emojis, or embedded media.

### Analysis categories

1. **Denotative description**
   - Describe only what is visibly present.
   - Include people without identifying unknown persons, objects, setting, actions frozen in the frame, text, graphics, interface elements, and embedded images/screens.
   - Distinguish observation from inference.

2. **Semiotic resources / modes**
   - Identify visible resources such as photographic image, illustration, writing, typography, colour, gesture/posture, spatial arrangement, symbols, diagrams, emojis, platform/interface elements, and image-within-image.
   - Note what each resource appears to contribute descriptively.

3. **Participants, processes, circumstances**
   - Participants: visible people, groups, objects, institutions represented by explicit text/logo, places, or other entities.
   - Processes: visible actions or represented processes.
   - Circumstances: visible spatial, temporal, environmental, or situational context.
   - Do not infer intentions or political roles unless explicitly stated in the frame.

4. **Composition and salience**
   - Foreground/background; centre/periphery; relative size/scale; camera distance/angle where observable; cropping; gaze/gesture direction; repetition; contrast; visual hierarchy.
   - Describe likely viewing order only when composition supports it.
   - Treat colour as a compositional resource, not as evidence of mood, ideology, nationality, or emotion unless explicit contextual evidence supports that reading.

5. **Salient signs / signifiers**
   - List especially prominent, repeated, foregrounded, or explicitly emphasized words, objects, symbols, gestures, colours, and graphic elements.
   - Keep them as descriptive signifiers. Do **not** assign Laclaudian status or political meaning.

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

### Output
Produce a compact structured description under the headings above, followed by:
- **Frame gist:** 1–3 neutral sentences.
- **Preserve for downstream analysis:** a short list of exact visible words/phrases and salient signs that later stages should receive unchanged where possible.
'''

    user_prompt = f'''
Analyze the provided frame using the social-semiotic pre-analysis categories above. Stay descriptive and modality-aware. Do not perform discourse or political analysis, and do not infer ideology, persuasion, populism, sentiment, or political alignment.
'''
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
    try:
        response = ollama.chat(model='llama3.2-vision:11b', 
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

def analyze_videos(language):
    """Analyze TikTok videos for a specific language."""
    filename = f'./csv/tiktok_{language}.csv'
    df = pd.read_csv(filename)
    df = df.dropna(subset=['whisperResult'])
    df = df.dropna(subset=['frame_files'])
    df['frame_analysis_1'] = ''
    df['frame_analysis_2'] = ''
    df['frame_analysis_3'] = ''
    df['frame_analysis_4'] = ''
    df['frame_analysis_5'] = ''
    df['frame_analysis_6'] = ''
    # Take only rows where language is the same
    df = df[df['language'] == language]
    for (index, row) in df.iterrows():
        author_username = row['authorUniqueId']
        video_id = row['videoId']
        # Check if exists in database
        c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
        if c.fetchone():
            logger.debug(f'Video already processed: {author_username} - {video_id}')
            # Get All from the database
            c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
            # Get all from the database
            row = c.fetchone()
            df.at[index, 'frame_analysis_1'] = str(row[2])
            df.at[index, 'frame_analysis_2'] = str(row[3])
            df.at[index, 'frame_analysis_3'] = str(row[4])
            df.at[index, 'frame_analysis_4'] = str(row[5])
            df.at[index, 'frame_analysis_5'] = str(row[6])
            df.at[index, 'frame_analysis_6'] = str(row[7])
        else:
            try:
                frame_files = row['frame_files']
                # Split frame_files
                frame_files = frame_files.split(',')
                frame_analysis_1 = ""
                frame_analysis_2 = ""
                frame_analysis_3 = ""
                frame_analysis_4 = ""
                frame_analysis_5 = ""
                frame_analysis_6 = ""
                frame_number = 1
                for i, frame_file in enumerate(frame_files):
                    frame_response = get_analysis(frame_file)
                    seconds = i * 30
                    frame_response = str(frame_response)
                    seconds = str(seconds)
                    frame_analysis = f'''### **Frame {frame_number} at {seconds} seconds**:                        
                    {frame_response}
                    '''
                    logger.debug(f'Frame analysis: {frame_analysis}')
                    if frame_number == 1:
                        frame_analysis_1 = str(frame_analysis)
                        df.at[index, 'frame_analysis_1'] = str(frame_analysis)
                    elif frame_number == 2:
                        frame_analysis_2 = str(frame_analysis)
                        df.at[index, 'frame_analysis_2'] = str(frame_analysis)
                    elif frame_number == 3:
                        frame_analysis_3 = str(frame_analysis)
                        df.at[index, 'frame_analysis_3'] = str(frame_analysis)
                    elif frame_number == 4:
                        frame_analysis_4 = str(frame_analysis)
                        df.at[index, 'frame_analysis_4'] = str(frame_analysis)
                    elif frame_number == 5:
                        frame_analysis_5 = str(frame_analysis)
                        df.at[index, 'frame_analysis_5'] = str(frame_analysis)
                    elif frame_number == 6:
                        frame_analysis_6 = str(frame_analysis)
                        df.at[index, 'frame_analysis_6'] = str(frame_analysis)
                    frame_number = frame_number + 1
                # Insert to database if not exists
                c.execute("INSERT INTO tiktok_videos (author_username, video_id, frame_analysis_1, frame_analysis_2, frame_analysis_3, frame_analysis_4, frame_analysis_5, frame_analysis_6) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",(str(author_username), str(video_id), str(frame_analysis_1), str(frame_analysis_2), str(frame_analysis_3), str(frame_analysis_4), str(frame_analysis_5), str(frame_analysis_6)))
                conn.commit()
            except Exception as e:
                logger.error(f'Error processing video: {e}')
    filename = f'./csv/tiktok_{language}.csv'
    df.to_csv(filename, index=False)


# Loop through all EP2024 TikTok languages and analyze videos
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']

for language in languages:
    analyze_videos(language)


c.close()
conn.close()


