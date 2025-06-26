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
    """Construct the user prompt for the Llama model."""
    user_message = f'''### **User Prompt**:

    **Data for Analysis**:

    1. **Multimodal Llama And EasyOCR Frame Analysis Results (1-6 Frames)**:
    ```
    {frame_analysis}
    ```

    2. **TikTok Metadata**:
    ```
    {metadata}
    ```
   
    3. **Whisper Transcript**:
    ```
    {transcript}
    ```
        
    ### **Task**:
    Utilize the provided data (**frame analysis**, **metadata** and **transcript**) to conduct a comprehensive political analysis of the TikTok video.
    '''
    return user_message

def get_llama_summary_system_prompt():
    """Construct the system prompt for the Llama model."""
    system_prompt = f'''### **System Prompt**:

    You are assisting a political scientist in analyzing a TikTok video related to the **2024 European Parliament elections**. 
    You are provided a **Whisper transcript**, **TikTok metadata**, **multimodal Llama frame analysis results for 1-6 frames** and **easyOCR results for each frame** to facilitate the analysis.
    Your role is to provide a structured and comprehensive political analysis using the multimodal frame analysis results, TikTok metadata, and Whisper transcript provided by the user.

    **Context**:
    The political scientist has supplied multimodal data, including multimodal video frame analysis, TikTok metadata, and Whisper transcript.
    Use this data to conduct a detailed political analysis of the TikTok video.

    **Instructions**:
    - Address each analysis category thoroughly by incorporating insights from the **multimodal Llama frame analysis results**, **TikTok metadata**, and **Whisper transcript**.
    - Ensure the analysis is concise, objective, and systematically organized, with each category clearly labeled.

    **Structure and Format**:
    - Present your analysis in a structured format, addressing each analysis category clearly and objectively.

    ### **Analysis Categories**:

    1. **Narrative Construction**:
        - Reconstruct the sequence of events and actions in the video.
        - Identify events and actions that shape the narrative of the video.

    2. **Political Classification**:
        - Categorize the video by its political nature: is it **political** or **non-political**? 
        - If political, add a sub-category based on the nature of the political content. 
        - Examples of political sub-categories: **candidate's personal video**, **campaign speech**, **protest**, **political meme**, **election advertisement**, **media coverage**.

    3. **Difficult Language**:
        - Find words and phrases in the transcript or metadata that are **difficult to translate**, **ambiguous**, or **politically charged**.
        - Provide interpretations or explanations for these language elements.
        - Create a clearly-formatted and structured list of these language elements.

    4. **Key Political Topics**:
        - Identify the major political topics in the video.
        - Examples of political topics: **immigration**, **climate change**, **populism**, **Ukraine war**, **Gaza conflict**.
        - Describe how these topics are presented in the video.
        - Create a clearly-formatted and structured list of these topics.

    5. **Political Entities**:
        - List political entities featured in the video.
        - Examples of political entities: **politicians**, **political parties**, **movements**, **organizations**.
        - Describe the role of these entities in the video.
        - Create a clearly-formatted and structured list of these entities.

    6. **Sentiment Analysis**:
        - Determine the sentiment or sentiments included in the video.
        - Classify the sentiment as **positive**, **negative**, or **neutral**. 
        - Identify the target of the sentiment (e.g., **the European Union**, **a political group**) and justify your evaluation.
        - Create a clearly-formatted and structured list of these sentiments.

    7. **Political Populism**:
        - Analyze the video for any populist elements using Ernesto Laclau’s theory of populism.
        - Identify **empty signifiers**, **chains of equivalence**, and the "people versus elite" narrative.
        - Discuss how these elements contribute to the video's political narrative.
        - Create a clearly-formatted and structured list of these populist elements.
        
    8. **Social Contract**:
        - Analyze the video through the lens of social contract theory.
        - Discuss any implied or explicit social agreements, obligations, or expectations between citizens and political authorities.
        - Explain how these social contracts shape political behavior.
        - Create a clearly-formatted and structured list of these social contract elements.
        
    9. **Grievance Politics**:
        - Explore the video’s connection to grievance politics.
        - Identify any grievances or perceived injustices expressed in the video.
        - Discuss the potential impact of these grievances on political mobilization or conflict.
        - Create a clearly-formatted and structured list of these grievances.
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


