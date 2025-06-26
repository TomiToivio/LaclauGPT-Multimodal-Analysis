import pandas as pd
import logging
from ollama import generate
from os import listdir
from os.path import isfile, join
import os
import cv2
import sqlite3
import easyocr
import whisper
from deep_translator import GoogleTranslator
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[RotatingFileHandler('./logs/preprocess.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# All EP2024 TikTok languages for OCR
reader = easyocr.Reader(['en','fr','pl','sv','pt','de','es','hu','hr']) 
# Load Whisper model
model = whisper.load_model('large', download_root='./whisper/')

# Sqlite3 database connection
conn = sqlite3.connect('./database/preprocess.db')
c = conn.cursor()
# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS tiktok_videos
                (author_username text, 
                video_id text,
                frames text,
                ocr_1 text,
                ocr_2 text,
                ocr_3 text,
                ocr_4 text,
                ocr_5 text,
                ocr_6 text,
                whisper_transcript text,
                whisper_language text,
                whisper_translated text)''')
conn.commit()

def save_keyframe(video_id, author_username, video_filename, frame_time, frame_number):
    """Extract and save a keyframe."""
    vidcap = cv2.VideoCapture(video_filename)
    milliseconds = frame_time * 1000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
    (success, image) = vidcap.read()
    directory = f'./Keyframes/TikTok/{author_username}/{video_id}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    new_filename = f'{directory}{frame_number}.jpg'
    if success:
        cv2.imwrite(new_filename, image)
        logger.debug(f'Keyframe saved for video {video_id}')
    return new_filename

def get_keyframes(video_filename, video_id, author_username):
    """Extract keyframes from a video file."""
    video = cv2.VideoCapture(video_filename)
    # Get the video duration
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    # Extract keyframes
    video.release()
    frame_files = []
    duration = int(duration)
    if duration > 180:
        duration = 180
    frame_number = 1
    for frame_time in range(0, duration, 30):
        frame_file = save_keyframe(video_id, author_username, video_filename, frame_time, frame_number)
        frame_files.append(frame_file)
        frame_number = frame_number + 1
    return frame_files

def get_transcript(video_id, author_username, scrapedCountry):
    """Get a Whisper transcript for a video."""
    # Video filename in CSC Allas 
    video_filename = f'./Allas/Scraper/TikTok/Videos/{scrapedCountry}/{author_username}/{video_id}.mp4'
    whisper_transcript = ''
    whisper_language = ''
    whisper_translated = ''
    try:
        result = model.transcribe(video_filename, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        whisper_transcript = str(result['text'])
        whisper_language = result['language']
        whisper_translated = GoogleTranslator(source=whisper_language, target='en').translate(whisper_transcript[:3000])
    except Exception as e:
        logger.error(f'Error transcribing video {video_id}: {e}')
        print(f'Error transcribing video {video_id}: {e}')
    return (whisper_transcript, whisper_language, whisper_translated)


def analyze_videos(language):
    """Preprocess TikTok videos for a specific language."""
    # We have exported the scraped TikTok data from MariaDB to a CSV file
    df = pd.read_csv(f'./csv/tiktok_videos.csv')
    df = df.dropna(subset=['whisperResult'])
    df['frame_files'] = ''
    df['ocr_1'] = ''
    df['ocr_2'] = ''
    df['ocr_3'] = ''
    df['ocr_4'] = ''
    df['ocr_5'] = ''
    df['ocr_6'] = ''
    df['whisper_transcript'] = ''
    df['whisper_language'] = ''
    df['whisper_translated'] = ''
    # Take only rows where the language is the same
    df = df[df['language'] == language]
    for (index, row) in df.iterrows():
        author_username = row['authorUniqueId']
        video_id = row['videoId']
        scrapedCountry = row['scrapedCountry']
        # CSC Allas video path
        video_path = f'./Allas/Scraper/TikTok/Videos/{scrapedCountry}/{author_username}/{video_id}.mp4'
        # Check if exists in database
        c.execute("SELECT * FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
        if c.fetchone():
            logger.debug(f'Video already processed: {author_username} - {video_id}')
            # Get the the video from the database
            c.execute("SELECT frames, ocr_1, ocr_2, ocr_3, ocr_4, ocr_5, ocr_6, whisper_transcript, whisper_language, whisper_translated FROM tiktok_videos WHERE author_username = ? AND video_id = ?", (str(author_username), str(video_id)))
            frames, ocr_1, ocr_2, ocr_3, ocr_4, ocr_5, ocr_6, whisper_transcript, whisper_language, whisper_translated = c.fetchone()
            df.at[index, 'frame_files'] = str(frames)
            df.at[index, 'ocr_1'] = str(ocr_1)
            df.at[index, 'ocr_2'] = str(ocr_2)
            df.at[index, 'ocr_3'] = str(ocr_3)
            df.at[index, 'ocr_4'] = str(ocr_4)
            df.at[index, 'ocr_5'] = str(ocr_5)
            df.at[index, 'ocr_6'] = str(ocr_6)
            df.at[index, 'whisper_transcript'] = str(whisper_transcript)
            df.at[index, 'whisper_language'] = str(whisper_language)
            df.at[index, 'whisper_translated'] = str(whisper_translated)
        else:
            # Check if video exists
            if not os.path.exists(video_path):
                logger.error(f'Video does not exist: {author_username} - {video_id}')
            else:
                try:
                    frame_files = get_keyframes(video_path, video_id, author_username)
                    # Split frame_files
                    frame_number = 1
                    ocr_1 = ''
                    ocr_2 = ''
                    ocr_3 = ''
                    ocr_4 = ''
                    ocr_5 = ''
                    ocr_6 = ''
                    for i, frame_file in enumerate(frame_files):
                        # OCR
                        results = reader.readtext(frame_file)
                        ocr_text = ''
                        for result in results:
                            ocr_text = ocr_text + result[1]
                            ocr_text = ocr_text + '\n'
                        if frame_number == 1:
                            ocr_1 = str(ocr_text)
                        elif frame_number == 2:
                            ocr_2 = str(ocr_text)
                        elif frame_number == 3:
                            ocr_3 = str(ocr_text)
                        elif frame_number == 4:
                            ocr_4 = str(ocr_text)
                        elif frame_number == 5:
                            ocr_5 = str(ocr_text)
                        elif frame_number == 6:
                            ocr_6 = str(ocr_text)
                        frame_number = frame_number + 1
                    # Get the whisper transcript
                    (whisper_transcript, whisper_language, whisper_translated) = get_transcript(video_id, author_username, scrapedCountry)
                    # Insert into database
                    c.execute("INSERT INTO tiktok_videos (author_username, video_id, frames, ocr_1, ocr_2, ocr_3, ocr_4, ocr_5, ocr_6, whisper_transcript, whisper_language, whisper_translated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (author_username, video_id, str(frame_files), str(ocr_1), str(ocr_2), str(ocr_3), str(ocr_4), str(ocr_5), str(ocr_6), str(whisper_transcript), str(whisper_language), str(whisper_translated)))
                    conn.commit()
                    # Frame files to string
                    frame_files = ','.join(frame_files)
                    # Add to dataframe
                    df.at[index, 'frame_files'] = str(frame_files)
                    df.at[index, 'ocr_1'] = str(ocr_1)
                    df.at[index, 'ocr_2'] = str(ocr_2)
                    df.at[index, 'ocr_3'] = str(ocr_3)
                    df.at[index, 'ocr_4'] = str(ocr_4)
                    df.at[index, 'ocr_5'] = str(ocr_5)
                    df.at[index, 'ocr_6'] = str(ocr_6)
                    df.at[index, 'whisper_transcript'] = str(whisper_transcript)
                    df.at[index, 'whisper_language'] = str(whisper_language)
                    df.at[index, 'whisper_translated'] = str(whisper_translated)
                except Exception as e:
                    logger.error(f'Error processing video: {e}')
    filename = f'./csv/tiktok_{language}.csv'
    df.to_csv(filename, index=False)

# All EP2024 TikTok languages for preprocessing
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']
for language in languages:
    analyze_videos(language)


c.close()
conn.close()

