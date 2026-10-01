import ast
import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler

import cv2
import easyocr
import pandas as pd
from deep_translator import GoogleTranslator

from asr_backend import describe_backend, load_asr_model
from ep24_video import (
    VIDEO_INITIAL_SKIP_SECONDS,
    analysis_frame_times,
    is_too_short,
    prepare_analysis_clip,
)

os.makedirs('./logs', exist_ok=True)
os.makedirs('./database', exist_ok=True)
os.makedirs('./analysis_clips', exist_ok=True)

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

reader = easyocr.Reader(['en', 'fr', 'pl', 'sv', 'pt', 'de', 'es', 'hu', 'hr'])
_asr = load_asr_model()
logger.info('ASR backend: %s', describe_backend())
logger.info(
    'EP24 media analysis starts at %.1fs to exclude the known initial scroll artifact',
    VIDEO_INITIAL_SKIP_SECONDS,
)

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
    """Return cached/new frame paths in the CSV format expected downstream."""
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


def best_transcript(legacy, transcript, translated):
    """Preserve legacy transcript data, filling gaps from new Whisper output."""
    for value in (legacy, transcript, translated):
        if value is not None:
            text = str(value).strip()
            if text and text.lower() != 'nan':
                return text
    return ''


def get_video_duration(video_filename):
    """Return video duration in seconds, or raise for unreadable media."""
    video = cv2.VideoCapture(video_filename)
    try:
        if not video.isOpened():
            raise ValueError(f'Could not open video: {video_filename}')
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or fps <= 0:
            raise ValueError(f'Invalid FPS ({fps}) for video: {video_filename}')
        return frame_count / fps
    finally:
        video.release()


def save_keyframe(video_id, author_username, video_filename, frame_time, frame_number):
    """Extract and save a keyframe, returning its path on success."""
    if frame_time < VIDEO_INITIAL_SKIP_SECONDS:
        raise ValueError(
            f'Frame time {frame_time} precedes mandatory EP24 analysis start '
            f'{VIDEO_INITIAL_SKIP_SECONDS}'
        )

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

        logger.debug('Keyframe saved for video %s at %.3fs', video_id, frame_time)
        return new_filename
    finally:
        vidcap.release()


def get_keyframes(video_filename, video_id, author_username):
    """Extract up to six keyframes, beginning after the known initial scroll."""
    duration = get_video_duration(video_filename)
    frame_files = []
    for frame_number, frame_time in enumerate(analysis_frame_times(duration), start=1):
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
    """Get a Whisper transcript after excluding the first-second scroll artifact."""
    video_filename = (
        f'./Allas/Scraper/TikTok/Videos/{scraped_country}/'
        f'{author_username}/{video_id}.mp4'
    )
    whisper_transcript = ''
    whisper_language = ''
    whisper_translated = ''

    try:
        duration = get_video_duration(video_filename)
        if is_too_short(duration):
            raise ValueError(
                f'video duration {duration:.3f}s is not longer than the mandatory '
                f'{VIDEO_INITIAL_SKIP_SECONDS:.1f}s initial skip'
            )

        analysis_clip = prepare_analysis_clip(video_filename, './analysis_clips')
        whisper_transcript, whisper_language = _asr(str(analysis_clip))

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
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        df = df.head(max_rows).copy()
        logger.info("Demo row limit active: processing first %s rows", max_rows)

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

    df['video_initial_skip_seconds'] = VIDEO_INITIAL_SKIP_SECONDS
    df['video_analysis_status'] = ''
    df['video_analysis_note'] = ''

    df = df[df['language'] == language].copy()

    for index, row in df.iterrows():
        author_username = row['authorUniqueId']
        video_id = row['videoId']
        scraped_country = row['scrapedCountry']
        legacy_whisper_result = row.get('whisperResult', '')
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
                legacy_whisper_result,
                whisper_transcript,
                whisper_translated,
            )
            df.at[index, 'video_analysis_status'] = 'cached'
            df.at[index, 'video_analysis_note'] = (
                'Historical cache preserved for reproducibility. New media analysis '
                f'uses t={VIDEO_INITIAL_SKIP_SECONDS:.1f}s onward.'
            )
            continue

        if not os.path.exists(video_path):
            logger.error('Video does not exist: %s - %s', author_username, video_id)
            df.at[index, 'video_analysis_status'] = 'missing_video'
            df.at[index, 'video_analysis_note'] = (
                'Source video was not available locally after Allas staging.'
            )
            continue

        try:
            duration = get_video_duration(video_path)
            if is_too_short(duration):
                message = (
                    f'Video is {duration:.3f}s long; no analyzable media remains '
                    f'after the mandatory {VIDEO_INITIAL_SKIP_SECONDS:.1f}s skip.'
                )
                logger.warning('%s - %s: %s', author_username, video_id, message)
                df.at[index, 'video_analysis_status'] = 'too_short'
                df.at[index, 'video_analysis_note'] = message
                continue

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
                legacy_whisper_result,
                whisper_transcript,
                whisper_translated,
            )
            df.at[index, 'video_analysis_status'] = 'ok'
            df.at[index, 'video_analysis_note'] = (
                f'Frames/OCR begin at t={VIDEO_INITIAL_SKIP_SECONDS:.1f}s; ASR uses '
                'a non-destructive derived clip with the same skip.'
            )
        except Exception as exc:
            logger.exception(
                'Error processing video %s - %s: %s',
                author_username,
                video_id,
                exc,
            )
            df.at[index, 'video_analysis_status'] = 'error'
            df.at[index, 'video_analysis_note'] = f'{type(exc).__name__}: {exc}'

    df.to_csv(f'./csv/tiktok_{language}.csv', index=False)


languages = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'en']
for language in languages:
    analyze_videos(language)

c.close()
conn.close()
