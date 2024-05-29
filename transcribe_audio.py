from groq import Groq
from openai import OpenAI
from pydub import AudioSegment
import os
from typing import List, Optional, Dict
import logging
import traceback
import shutil
from dotenv import load_dotenv
import time
import httpx
import json

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def call_groq(audio_file: str) -> str:
    """
    This function transcribes an audio file using the Groq library.

    Args:
        audio_file (str): The path to the audio file to be transcribed.

    Returns:
        str: JSON with the transcribed audio.
    """
    # client = Groq()
    client = OpenAI()
    max_retries = 6
    retry_delay = 10  # seconds

    # Transcribe the audio file
    try:
        # Get the file size in megabytes and log it
        file_size_bytes = os.path.getsize(audio_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        logging.info(f"The size of the file sent to Groq Whisper is {file_size_megabytes} MB.")

        for attempt in range(max_retries):
            try:
                with open(audio_file, "rb") as af:
                    transcription= client.audio.transcriptions.create(
                        file=(audio_file, af.read()),
                        # model="whisper-large-v3",
                        model="whisper-1",
                        response_format="verbose_json"
                    )
                time.sleep(3)
                return transcription
            except Exception as e:
                if "429" in str(e) or "rate_limit_exceeded" in str(e):
                    logging.info(f"Received 429 response, waiting {retry_delay} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Unexpected error occurred: {e}")
                    if attempt == max_retries - 1:
                        logging.error("Max retries reached, failed to transcribe audio.")
                        return None
                    time.sleep(retry_delay)
    except FileNotFoundError:
        logging.error(f"File {audio_file} not found.")
        raise
    except httpx.ConnectError as e:
        logging.error(f"Connection error occurred: {e}")
        raise
    except httpx.TimeoutException as e:
        logging.error(f"Request timeout: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        raise

def create_audio_chunks(audio_file: str, temp_dir: str, chunk_size: int=25*60000) -> List[str]:
    """
    Splits an audio file into smaller segments or chunks based on a specified duration. This function is useful for processing large audio files incrementally or in parallel, which can be beneficial for tasks such as audio analysis or transcription where handling smaller segments might be more manageable.

    AudioSegment can slice an audio file by specifying the start and end times in milliseconds. This allows you to extract precise segments of the audio without needing to process the entire file at once. For example, `audio[1000:2000]` extracts a segment from the 1-second mark to the 2-second mark of the audio file.

    Args:
        audio_file (str): The absolute or relative path to the audio file that needs to be chunked. This file should be accessible and readable.
        
        chunk_size (int): The length of each audio chunk expressed in milliseconds. This value determines how the audio file will be divided. For example, a `chunk_size` of 1000 milliseconds will split the audio into chunks of 1 second each.
        
        temp_dir (str): The directory where the temporary audio chunk files will be stored. This directory will be used to save the output chunk files, and it must have write permissions. If the directory does not exist, it will be created.

    Returns:
        List[str]: A list containing the file paths of all the audio chunks created. Each path in the list represents a single chunk file stored in the specified `temp_dir`. The files are named sequentially based on their order in the original audio file.

    Raises:
        FileNotFoundError: If the `audio_file` does not exist or is inaccessible.
        
        PermissionError: If the script lacks the necessary permissions to read the `audio_file` or write to the `temp_dir`.
        ValueError: If `chunk_size` is set to a non-positive value.
    """
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        logging.error(f"create_audio_chunks failed to load audio file {audio_file}: {e}")
        logging.error(traceback.format_exc())
        return []

    start = 0
    end = chunk_size
    counter = 0
    chunk_files = []

    while start < len(audio):
        chunk = audio[start:end]
        chunk_file_path = os.path.join(temp_dir, f"{counter}_{file_name}.mp3")
        try:
            chunk.export(chunk_file_path, format="mp3") # Using .mp3 because it's cheaper
            chunk_files.append(chunk_file_path)
        except Exception as e:
            error_message = f"create_audio_chunks failed to export chunk {counter}: {e}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            raise error_message
        start += chunk_size
        end += chunk_size
        counter += 1
    return chunk_files

def clump_response(transcription: str, added_duration: float=0.0) -> List[dict]:
    if isinstance(transcription, list):
        transcription = json.dumps(transcription)

    segments = []
    current_segment = ""
    for segment in transcription.segments:
        if current_segment == "":
            current_segment += segment["text"]
            start_time = segment["start"]
        elif segment["text"].strip().endswith(('.', '?', '!')):
            current_segment += segment["text"]
            end_time = segment["end"]
            segments.append({
                "start_time": start_time + added_duration, 
                "end_time": end_time + added_duration, 
                "text": current_segment
            })
            current_segment = ""
        else:
            current_segment += segment["text"]

    return segments

def default_response(transcription: str, added_duration: float=0.0) -> List[dict]:
    if isinstance(transcription, list):
        transcription = json.dumps(transcription)

    segments = []
    for segment in transcription.segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        segments.append({
                "start_time": start_time + added_duration, 
                "end_time": end_time + added_duration, 
                "text": text
        })

    return segments

def transcribe_chunks(audio_file: str, temp_dir: str, chunk_size: int=25*60000, response_type: str="default") -> str:
    try:
        chunk_files = create_audio_chunks(audio_file, temp_dir, chunk_size)
    except Exception as e:
        logging.error(f"Failed to create audio chunks: {e}")
        return []
    duration = 0
    all_segments = []
    for chunk_file in chunk_files:
        response = call_groq(chunk_file)
        if response_type == "clump":
            formatted_response = clump_response(response, duration)
        else:
            formatted_response = default_response(response, duration)
        all_segments.extend(formatted_response)
        duration += response.duration
    return all_segments

if __name__ == "__main__":
    from download_video import yt_dlp_download
    url = "https://www.youtube.com/watch?v=9l0ieOqLMxk&ab_channel=CNBCTelevision"
    audio_file = yt_dlp_download(url)
    segments = transcribe_chunks(audio_file, "temp")
    output_file = "segments.json"
    with open(output_file, "w") as f:
        json.dump(segments, f, indent=4)