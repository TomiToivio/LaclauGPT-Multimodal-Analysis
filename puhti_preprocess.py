import ast
import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler

import cv2
import easyocr
import pandas as pd
import whisper
from deep_translator import GoogleTranslator

# The repository does not contain runtime directories, so create them before
# constructing file handlers or SQLite connections.
os.makedirs('./logs', exist_ok=True)
os.makedirs('./database', exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        RotatingFileHandler(
            './logs/preprocess.log',
            encoding='utf-8',
            maxBytes=1000000,
            backupCount=5,
        )
    ],
    level=logging.DEBUG,
)

# All EP2024 TikTok languages used for OCR.
reader = easyocr.Reader(['en', 'fr', 'pl', 'sv', 'pt', 'de', 'es', 'hu', 'hr'])

# Load Whisper model.
model = whisper.load_model('large', download_root='./whisper/')

# SQLite database connection.
conn = sqlite3.connect('./database/preprocess.db')
c = conn.cursor()
c.execute(
    '''CREATE TABLE IF NOT EXISTS tiktok_videos
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
        whisper_translated text)'''
)
conn.commit()


def normalize_frame_files(value):
    """Return cached/new frame paths in the CSV format expected downstream.

    Older cache rows stored ``str(list)`` while fresh rows used comma-separated
    paths. Accept both representations so old preprocessing databases continue
    to work with ``puhti_frame.py``.
    """
    if value is None:
        return ''

    if isinstance(value, (list, tuple)):
        frame_files = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() == 'nan':
            return ''
        try:
            parsed = ast.literal_eval(text)
            frame_files = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
        except (ValueError, SyntaxError):
            frame_files = text.split(',')

    cleaned = []
    for frame_file in frame_files:
        path = str(frame_file).strip().strip('[]').strip().strip("'\"")
        if path:
            cleaned.append(path)
    return ','.join(cleaned)


def best_transcript(transcript, translated):
    """Prefer English translation while keeping the original as a fallback."""
    for value in (translated, transcript):
        if value is not None:
            text = str(value).strip()
            if text and text.lower() != 'nan':
                return text
    return ''


def save_keyframe(video_id, author_username, video_filename, frame_time, frame_number):
    """Extract and save a keyframe, returning its path on success."""
    vidcap = cv2.VideoCapture(video_filename)
    try:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        success, image = vidcap.read()
        if not success or image is None:
            logger.warning(
                'Could not read frame %s at %ss from video %s',
                frame_number,
                frame_time,
                video_id,
            )
            return None

        directory = f'./Keyframes/TikTok/{author_username}/{video_id}/'
        os.makedirs(directory, exist_ok=True)
        new_filename = f'{directory}{frame_number}.jpg'
        if not cv2.imwrite(new_filename, image):
            logger.warning('Could not write keyframe %s for video %s', frame_number, video_id)
            return None

        logger.debug('Keyframe saved for video %s', video_id)
        return new_filename
    finally:
        vidcap.release()


def get_keyframes(video_filename, video_id, author_username):
    """Extract up to six keyframes, one every 30 seconds for 180 seconds."""
    video = cv2.VideoCapture(video_filename)
    try:
        if not video.isOpened():
            raise ValueError(f'Could not open video: {video_filename}')

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or fps <= 0:
            raise ValueError(f'Invalid FPS ({fps}) for video: {video_filename}')

        duration = frame_count / fps
    finally:
        video.release()

    # Always try the first frame for a valid very short video.
    duration_seconds = max(1, min(int(duration), 180))
    frame_files = []
    for frame_number, frame_time in enumerate(range(0, duration_seconds, 30), start=1):
        frame_file = save_keyframe(
            video_id,
            author_username,
            video_filename,
            frame_time,
            frame_number,
        )
        if frame_file:
            frame_files.append(frame_file)
    return frame_files


def get_transcript(video_id, author_username, scraped_country):
    """Get a Whisper transcript and, when useful, an English translation."""
    video_filename = (
        f'./Allas/Scraper/TikTok/Videos/{scraped_country}/'
        f'{author_username}/{video_id}.mp4'
    )
    whisper_transcript = ''
    whisper_language = ''
    whisper_translated = ''

    try:
        result = model.transcribe(
            video_filename,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        whisper_transcript = (
            ' '.join(result['text'])
            if isinstance(result['text'], list)
            else str(result['text'])
        )
        whisper_language = str(result.get('language', ''))

        if whisper_transcript:
            if whisper_language == 'en':
                whisper_translated = whisper_transcript
            elif whisper_language:
                whisper_translated = GoogleTranslator(
                    source=whisper_language,
                    target='en',
                ).translate(whisper_transcript[:3000])
    except Exception as exc:
        logger.error('Error transcribing video %s: %s', video_id, exc)

    return whisper_transcript, whisper_language, whisper_translated


def analyze_videos(language):
    """Preprocess TikTok videos for a specific language."""
    df = pd.read_csv('./csv/tiktok_videos.csv')

    # ``whisperResult`` is a legacy downstream field. Do not drop rows based on
    # it before preprocessing, because this stage is the one that creates the
    # Whisper transcript in the first place.
    if 'whisperResult' not in df.columns:
        df['whisperResult'] = ''

    for column in (
        'frame_files',
        'ocr_1',
        'ocr_2',
        'ocr_3',
        'ocr_4',
        'ocr_5',
        'ocr_6',
        'whisper_transcript',
        'whisper_language',
        'whisper_translated',
    ):
        df[column] = ''

    df = df[df['language'] == language].copy()

    for index, row in df.iterrows():
        author_username = row['authorUniqueId']
        video_id = row['videoId']
        scraped_country = row['scrapedCountry']
        video_path = (
            f'./Allas/Scraper/TikTok/Videos/{scraped_country}/'
            f'{author_username}/{video_id}.mp4'
        )

        c.execute(
            'SELECT frames, ocr_1, ocr_2, ocr_3, ocr_4, ocr_5, ocr_6, '
            'whisper_transcript, whisper_language, whisper_translated '
            'FROM tiktok_videos WHERE author_username = ? AND video_id = ?',
            (str(author_username), str(video_id)),
        )
        cached = c.fetchone()

        if cached:
            logger.debug('Video already processed: %s - %s', author_username, video_id)
            (
                frames,
                ocr_1,
                ocr_2,
                ocr_3,
                ocr_4,
                ocr_5,
                ocr_6,
                whisper_transcript,
                whisper_language,
                whisper_translated,
            ) = cached

            df.at[index, 'frame_files'] = normalize_frame_files(frames)
            df.at[index, 'ocr_1'] = str(ocr_1 or '')
            df.at[index, 'ocr_2'] = str(ocr_2 or '')
            df.at[index, 'ocr_3'] = str(ocr_3 or '')
            df.at[index, 'ocr_4'] = str(ocr_4 or '')
            df.at[index, 'ocr_5'] = str(ocr_5 or '')
            df.at[index, 'ocr_6'] = str(ocr_6 or '')
            df.at[index, 'whisper_transcript'] = str(whisper_transcript or '')
            df.at[index, 'whisper_language'] = str(whisper_language or '')
            df.at[index, 'whisper_translated'] = str(whisper_translated or '')
            df.at[index, 'whisperResult'] = best_transcript(
                whisper_transcript,
                whisper_translated,
            )
            continue

        if not os.path.exists(video_path):
            logger.error('Video does not exist: %s - %s', author_username, video_id)
            continue

        try:
            frame_files = get_keyframes(video_path, video_id, author_username)
            ocr_values = [''] * 6

            for i, frame_file in enumerate(frame_files[:6]):
                results = reader.readtext(frame_file)
                ocr_values[i] = '\n'.join(str(result[1]) for result in results)

            (
                whisper_transcript,
                whisper_language,
                whisper_translated,
            ) = get_transcript(video_id, author_username, scraped_country)

            serialized_frames = normalize_frame_files(frame_files)
            c.execute(
                'INSERT INTO tiktok_videos '
                '(author_username, video_id, frames, ocr_1, ocr_2, ocr_3, '
                'ocr_4, ocr_5, ocr_6, whisper_transcript, whisper_language, '
                'whisper_translated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (
                    str(author_username),
                    str(video_id),
                    serialized_frames,
                    *ocr_values,
                    str(whisper_transcript),
                    str(whisper_language),
                    str(whisper_translated),
                ),
            )
            conn.commit()

            df.at[index, 'frame_files'] = serialized_frames
            for i, value in enumerate(ocr_values, start=1):
                df.at[index, f'ocr_{i}'] = value
            df.at[index, 'whisper_transcript'] = str(whisper_transcript)
            df.at[index, 'whisper_language'] = str(whisper_language)
            df.at[index, 'whisper_translated'] = str(whisper_translated)
            df.at[index, 'whisperResult'] = best_transcript(
                whisper_transcript,
                whisper_translated,
            )
        except Exception as exc:
            logger.exception(
                'Error processing video %s - %s: %s',
                author_username,
                video_id,
                exc,
            )

    df.to_csv(f'./csv/tiktok_{language}.csv', index=False)


# All EP2024 TikTok languages for preprocessing.
languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']
for language in languages:
    analyze_videos(language)

c.close()
conn.close()
