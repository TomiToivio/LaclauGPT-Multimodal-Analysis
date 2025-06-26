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
    # System prompt with instructions for detailed frame analysis
    system_prompt = f'''### **System Prompt**

    You are a political scientist analyzing a single frame from a TikTok video concerning the 2024 European Parliament elections.

    **Provided Data**:
    - **Video Frame**: One frame from the TikTok video.

    ### **Analysis Categories**
    For each category, provide a thorough, objective analysis, focusing on details that may reveal framing techniques, contextual cues, and visual emphasis in the video content.

    1. **Framing**:
    - Identify types of shots, such as close-ups of politicians (which may emphasize importance) or wide-angle shots of crowds and public spaces.
    - Note any framing choices that highlight objects or gestures (e.g., raised hands).
    - Observe split-screen layouts (dual images within the frame).

    2. **Visual Elements**:
    - Describe the background context, noting features like public squares, government buildings, natural landscapes, vehicles, campaign events, flags, or office interiors.
    - Specify whether the scene is set outdoors or indoors, or in a studio environment.

    3. **Activity**:
    - Identify visible activities, such as politicians giving speeches, demonstrations, or scenes that indicate voter participation.

    4. **Color Scheme**:
    - Analyze the color palette, considering how it might evoke a European vs. national context or convey mood.

    5. **Objects**:
    - Note prominent objects such as campaign posters, ballots, microphones, national or EU flags, signs, podiums, or digital graphics.
    - Identify minor items like coffee mugs, on-screen text, emojis, or secondary images (e.g., “image-in-image” features).

    6. **Subjects**:
    - Identify visible individuals or groups, including politicians, influencers, campaigners, voters, activists, or citizens.
    - Note any appearances of pets.

    7. **Screen Recording Indicators**:
    - Observe if the frame includes content from TV, YouTube, or other social media, or shows people filming something on another screen.
    '''
    user_prompt = f'''
    Analyze the provided video frame based on the categories outlined in the system prompt. Provide a detailed description of the visual elements, activities, and subjects present in the frame. Focus on how these elements contribute to the overall message or framing of the video content.
    '''
    logger.debug(f'Processing image: {frame_file}')
    images = []
    with open(frame_file, 'rb') as f:
        raw = f.read()
        raw = base64.b64encode(raw)
        images.append(raw)
    # Temperature 0.0 was found to be the best for this task
    options={"repeat_last_n": 64,
             "repeat_penalty": 1.1,
             "num_ctx": 8096,
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


