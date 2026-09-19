import pandas as pd
import logging
from ollama import generate
from os import listdir
from os.path import isfile, join
import os
import cv2
import ollama
import base64
import sqlite3
import time
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
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

def get_llama_summary_user_prompt(metadata, transcript, frame_analysis):
    """Construct the user prompt for social-semiotic multimodal pre-analysis."""
    user_message = f'''### User Prompt

### Input data

1. **Sampled frame analyses**
```
{frame_analysis}
```

2. **Source/platform metadata**
```
{metadata}
```

3. **Speech / transcript**
```
{transcript}
```

### Task

Integrate all available modalities into one **descriptive multimodal social-semiotic first pass**.

Treat frame descriptions, written/visible text, transcript, metadata, and temporal sequence as distinct evidence streams. Preserve disagreements between them rather than forcing a single interpretation. Metadata can provide context but must not override what is actually present in the media.

Do not perform political, ideological, partisan, populism, sentiment, discourse, or Laclauian analysis. Do not classify empty/floating signifiers, nodal points, chains of equivalence, antagonisms, hegemony, political camps, motives, or persuasive effectiveness. Those belong to downstream analysis.
'''
    return user_message


def get_llama_summary_system_prompt():
    """Construct the system prompt for multimodal social-semiotic pre-analysis."""
    system_prompt = f'''### System Prompt

You are assisting a social-science research pipeline by creating a **Multimodal Social-Semiotic Pre-Analysis** of incoming video/image/text material.

The methodological orientation is:
- social semiotics and multimodality: signs and semiotic resources make meaning across modes;
- Halliday/SFL: attend descriptively to textual organisation, represented participants/processes/circumstances, and relations between communicator/content/audience when directly observable;
- Kress & van Leeuwen: composition, salience, vectors, conceptual vs narrative visual structures, and affordances of modes;
- structuralist preparation: preserve salient signifiers, contrasts, co-occurrences, and relations for later analysis;
- a cautious denotation/connotation distinction: describe what is present first, then record only well-supported culturally available associations.

This is explicitly **before discourse analysis**. The purpose is to transform heterogeneous media into a faithful, structured account of signs and cross-modal relations that downstream LaclauGPT stages can analyze.

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

11. **Downstream-preservation block**
   - Exact salient words/phrases/hashtags.
   - Named entities explicitly present in source material.
   - Recurring visual/symbolic elements.
   - Important cross-modal contrasts or associations.
   - Do not interpret these items politically.

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
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug(f"User prompt: {user_prompt}")
    response = ollama.chat(model="llama3.2-vision:11b", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
    ], options=options)
    llama_response = response['message']['content']
    logger.debug(f"LLAMA response: {llama_response}")
    return llama_response


def analyze_videos(language):
    """Analyze TikTok videos for a specific language."""
    # Read csv
    filename = f'./csv/tiktok_{language}.csv'
    df = pd.read_csv(filename)
    df = df.dropna(subset=['whisperResult'])
    df['summary_analysis'] = ''
    # Take only rows where language is fi
    for (index, row) in df.iterrows():
        author_username = row['authorUniqueId']
        video_id = row['videoId']
        logger.debug(f'Analyzing video {row["videoId"]}')
        video_timestamp = row['videoCreated']
        video_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(video_timestamp))
        video_duration = row['videoDuration']
        video_diggcount = row['videoDiggCount']
        video_sharecount = row['videoShareCount']
        video_commentcount = row['videoCommentCount']
        video_playcount = row['videoPlayCount']
        video_description = row['videoDescription']
        author_name = row['authorNickname']
        author_signature = row['authorSignature']
        video_url = f'https://www.tiktok.com/@{author_username}/video/{video_id}'
        author_url = f'https://www.tiktok.com/@{author_username}'
        hashtags = ''
        video_description = str(video_description)
        try:
            hashtags = [tag.strip() for tag in video_description.split() if tag.startswith('#')]
            hashtags = ', '.join(hashtags)
        except Exception as e:
            logger.error(f'Error extracting hashtags: {e}')
            hashtags = ''
        metadata = f'''- Author name: {author_name}        
        - Author username: {author_username}        
        - Author signature: {author_signature}        
        - Description: {video_description}      
        - Timestamp: {video_timestamp}      
        - Duration: {video_duration}        
        - Diggs: {video_diggcount}        
        - Shares: {video_sharecount}        
        - Comments: {video_commentcount}        
        - Plays: {video_playcount}    
        - Video URL: {video_url}        
        - Author URL: {author_url} 
        - Hashtags: {hashtags}
        '''
        logger.debug(f'Metadata: {metadata}')
        transcript = row['whisperResult']
        logger.debug(f'Transcript: {transcript}')
        # Add metadata to row
        df.at[index, 'metadata'] = str(metadata)
        # Create frame analysis
        frame_analysis_1 = row['frame_analysis_1']
        frame_analysis_2 = row['frame_analysis_2']
        frame_analysis_3 = row['frame_analysis_3']
        frame_analysis_4 = row['frame_analysis_4']
        frame_analysis_5 = row['frame_analysis_5']
        frame_analysis_6 = row['frame_analysis_6']
        ocr_1 = row['ocr_1']
        ocr_2 = row['ocr_2']
        ocr_3 = row['ocr_3']
        ocr_4 = row['ocr_4']
        ocr_5 = row['ocr_5']
        ocr_6 = row['ocr_6']
        frame_analysis = f'''

        {frame_analysis_1}

        ### OCR results for frame 1 at 0 seconds:
        
        {ocr_1}
            
        '''
            
        # If frame_analysis_2 exists and not empty string
        if frame_analysis_2:
            frame_analysis = frame_analysis + f'''
            
            {frame_analysis_2}
                        
            ### OCR results for frame 2 at 30 seconds:
            
            {ocr_2}
            
            '''
        
        # If frame_analysis_3 exists and not empty string
        if frame_analysis_3:
            frame_analysis = frame_analysis + f'''
            
            {frame_analysis_3}
                        
            ### OCR results for frame 3 at 60 seconds:
            
            {ocr_3}
            
            '''
            
        # If frame_analysis_4 exists and not empty string
        if frame_analysis_4:
            frame_analysis = frame_analysis + f'''
            
            {frame_analysis_4}
                        
            ### OCR results for frame 4 at 90 seconds:
            
            {ocr_4}
            
            '''
            
        # If frame_analysis_5 exists and not empty string
        if frame_analysis_5:
            frame_analysis = frame_analysis + f'''
            
            {frame_analysis_5}
                        
            ### OCR results for frame 5 at 120 seconds:
            
            {ocr_5}
            
            '''
            
        # If frame_analysis_6 exists and not empty string
        if frame_analysis_6:
            frame_analysis = frame_analysis + f'''
            
            {frame_analysis_6}
                        
            ### OCR results for frame 6 at 150 seconds:
            
            {ocr_6}
            
            '''
        
        logger.debug(f'Frame analysis: {frame_analysis}')
        
        # Check if exists in database
        c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
        if c.fetchone():
            logger.debug(f'Video already processed: {author_username} - {video_id}')
            # Get the frame analysis from the database
            c.execute("SELECT summary_analysis FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
            summary_analysis = c.fetchone()[0]
            df.at[index, 'summary_analysis'] = str(summary_analysis)
        else:
            try:
                user_prompt = get_llama_summary_user_prompt(metadata, transcript, frame_analysis)
                system_prompt = get_llama_summary_system_prompt()
                summary_analysis = get_llama_summary_response(system_prompt, user_prompt)
                c.execute("INSERT INTO tiktok_videos (author_username, video_id, summary_analysis) VALUES (?, ?, ?)",(str(author_username), str(video_id), str(summary_analysis)))
                conn.commit()
                df.at[index, 'summary_analysis'] = str(summary_analysis)
                logger.debug(f'Summary analysis: {summary_analysis}')
            except Exception as e:
                logger.error(f'Error processing video: {e}')
    df.to_csv(f'./csv/tiktok_{language}.csv', index=False)

# Loop through each EP2024 TikTok language and analyze videos
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']
for language in languages:
    analyze_videos(language)

c.close()
conn.close()


