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

def create_audio_chunks(audio_file_path: str, temp_dir: str, chunk_size: int=25*60000) -> List[str]:
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
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as e:
        logging.error(f"create_audio_chunks failed to load audio file {audio_file_path}: {e}")
        logging.error(traceback.format_exc())
        return []

    start = 0
    end = chunk_size
    counter = 0
    audio_chunk_paths = []

    while start < len(audio):
        chunk = audio[start:end]
        chunk_file_path = os.path.join(temp_dir, f"{counter}_{file_name}.mp3")
        try:
            chunk.export(chunk_file_path, format="mp3") # Using .mp3 because it's cheaper
            audio_chunk_paths.append(chunk_file_path)
        except Exception as e:
            error_message = f"create_audio_chunks failed to export chunk {counter}: {e}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            raise error_message
        start += chunk_size
        end += chunk_size
        counter += 1
    return audio_chunk_paths

def transcribe_chunks(audio_chunk_paths: List[str]) -> List[dict]:

    transcribed_chunks = []

    for chunk_path in audio_chunk_paths:
        try:
            response = call_groq(chunk_path)
        except Exception as e:
            logging.error(f"transcribe_chunks failed to call_groq for {chunk_path}: {e}")
            response = None

        transcribed_chunk = {
            "segments": response.segments,
            "duration": response.duration
        }
        transcribed_chunks.append(transcribed_chunk)
    
    return transcribed_chunks

def clump_response(segments: str, added_duration: float=0.0) -> List[dict]:

    formatted_segments = []
    current_segment = ""
    current_segments = []
    start_time = 0
    end_time = 0

    for segment in segments:
        # If the current segment is empty, add the text to the current segment
        if current_segment == "":
            current_segment += segment["text"]
            current_segments.append(segment)
            start_time = segment["start"]
        # If the current segment ends with a punctuation mark, add the current segment to the formatted segments
        elif segment["text"].strip().endswith(('.', '?', '!')) or segment["end"] == segments[-1]["end"]:
            end_time = segment["end"]
            if end_time - start_time > 60:
                for s in current_segments:
                    formatted_segments.append({
                        "start_time": s["start"] + added_duration,
                        "end_time": s["end"] + added_duration,
                        "text": s["text"]
                    })
            else:
                current_segment += segment["text"]
                formatted_segments.append({
                    "start_time": start_time + added_duration, 
                    "end_time": end_time + added_duration, 
                    "text": current_segment
                })
            current_segment = ""
            current_segments = []
        else:
            current_segment += segment["text"]
            current_segments.append(segment)

    return formatted_segments

def default_response(segments: str, added_duration: float=0.0) -> List[dict]:
    
    formatted_segments = []
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        formatted_segments.append({
                "start_time": start_time + added_duration, 
                "end_time": end_time + added_duration, 
                "text": text
        })

    return formatted_segments

def format_chunks(transcribed_chunks: List[dict], response_type: str="default") -> List[dict]:
    
    duration = 0
    segments = []

    for chunk_dict in transcribed_chunks:
        if response_type == "clump":
            formatted_response = clump_response(chunk_dict["segments"], duration)
        else:
            formatted_response = default_response(chunk_dict["segments"], duration)
        segments.extend(formatted_response)
        duration += chunk_dict["duration"]

    return segments

def main_transcribe_audio(audio_file_path: str, response_type: str="clump"):
    
    audio_chunks = create_audio_chunks(audio_file_path, "temp")
    transcribed_chunks = transcribe_chunks(audio_chunks)
    segments = format_chunks(transcribed_chunks, response_type=response_type)
    
    return segments



if __name__ == "__main__":
    new_create = True
    from download_video import yt_dlp_download
    url = "https://www.youtube.com/watch?v=wz6-0EPBvqE&t=275s&ab_channel=IndianaPacers"
    
    if new_create is False:
        audio_file_path = yt_dlp_download(url)
        with open("audio_file_path.txt", "w") as f:
            f.write(audio_file_path)
    else:
        with open("audio_file_path.txt", "r") as f:
            audio_file_path = f.read()
    print(f"Audio file path: {audio_file_path}")

    if new_create is False:
        audio_chunks = create_audio_chunks(audio_file_path, "temp")
        transcribed_chunks = transcribe_chunks(audio_chunks)
        output_file = "transcribed_chunks.json"
        import json
        with open(output_file, "w") as f:
            json.dump(transcribed_chunks, f, indent=4)
    else:
        with open("transcribed_chunks.json", "r") as f:
            transcribed_chunks = json.load(f)
            
    
    if new_create is True:
        segments = format_chunks(transcribed_chunks, response_type="clump")
        output_file = "segments.json"
        with open(output_file, "w") as f:
            json.dump(segments, f, indent=4)
    else:
        with open("segments.json", "r") as f:
            segments = json.load(f)
            


