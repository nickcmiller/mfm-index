import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from groq import Groq

from typing import List, Optional, Dict, Any
import logging
import traceback
import shutil
from dotenv import load_dotenv
import time
import httpx
import json


logging.basicConfig(level=logging.INFO)
load_dotenv(os.path.join(root_dir, '.env'))

def whisper_call(
    audio_file: str,
    transcription_service: str = "Groq",
    max_retries: int = 6,
    retry_delay: int = 3
) -> Dict[str, Any]:
    """
        This function transcribes an audio file using either Groq or OpenAI.

        Args:
            audio_file (str): The path to the audio file to be transcribed.
            transcription_service (str): The transcription service to use, either "Groq" or "OpenAI".
            max_retries (int): The maximum number of retries to attempt if the transcription fails.
            retry_delay (int): The delay between retries in seconds.

        Returns:
            Dict[str, Any]: JSON with the transcribed audio.

        Example:
            >>> transcription = whisper_call("audio.mp3", "Groq")
            >>> print(transcription)
            {
                "task": "transcribe",
                "language": "english",
                "duration": 3.2,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 3.2,
                        "text": "Hello, World!",
                        "tokens": [1, 2, 3, 4, 5],
                        "avg_logprob": -0.5,
                        "compression_ratio": 0.8
                    }
                ],
                "text": "Hello, World!"
            }
    """
    # Validate input parameters
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"The specified audio file does not exist: {audio_file}")
    if transcription_service not in ["Groq", "OpenAI"]:
        raise ValueError("Invalid transcription service specified. Choose 'Groq' or 'OpenAI'.")

    # Initialize client based on the transcription service
    if transcription_service == "Groq":
        client = Groq()
        whisper_model = "whisper-large-v3"
    else:
        from openai import OpenAI
        client = OpenAI()
        whisper_model = "whisper-1"

    try:
        file_size_bytes = os.path.getsize(audio_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        logging.info(f"Transcribing {file_size_megabytes:.2f} MB file using {transcription_service}.")

        for attempt in range(max_retries):
            try:
                with open(audio_file, "rb") as af:
                    transcription = client.audio.transcriptions.create(
                        file=(audio_file, af.read()),
                        model=whisper_model,
                        response_format="verbose_json"
                    )
                return transcription
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if "429" in str(e) or "rate_limit_exceeded" in str(e):
                    logging.info("Rate limit exceeded, retrying after delay.")
                    time.sleep(retry_delay)
                elif attempt == max_retries - 1:
                    logging.error("Max retries reached, transcription failed.")
                    return None
                else:
                    retry_delay *= 2
    except Exception as e:
        logging.error(f"Failed to transcribe audio file: {e}")
        raise


def transcribe_chunks(
    audio_chunk_paths: List[str], 
    temp_dir: str
) -> List[dict]:
    """
        Transcribes a list of audio chunks and returns the transcription results.

        This function takes a list of file paths to audio chunks and a temporary directory path. It iterates over each audio chunk, calls the `call_groq` function to transcribe the chunk, and appends the transcription result to the `transcribed_chunks` list. 

        The transcription result for each chunk is a dictionary containing the following keys:
        - "segments": The transcribed segments returned by the `call_groq` function.
        - "duration": The duration of the audio chunk.

        If an exception occurs during the transcription process for a chunk, an error message is logged along with the traceback.

        Args:
            audio_chunk_paths (List[str]): A list of file paths to the audio chunks to be transcribed.
            temp_dir (str): The path to the temporary directory where the audio chunks are stored.

        Returns:
            List[dict]: A list of dictionaries representing the transcription results for each audio chunk. Each dictionary contains the "segments" and "duration" keys.

        Example:
            >>> audio_chunk_paths = ["chunk1.mp3", "chunk2.mp3", "chunk3.mp3"]
            >>> temp_dir = "/tmp/audio_chunks"
            >>> transcribed_chunks = transcribe_chunks(audio_chunk_paths, temp_dir)
            >>> print(transcribed_chunks)
            [
                {
                    "segments": [{"text": "Hello world", "start": 0.0, "end": 1.5}, ...],
                    "duration": 25.0
                },
                {
                    "segments": [{"text": "This is the second chunk", "start": 0.0, "end": 2.3}, ...],
                    "duration": 25.0
                },
                {
                    "segments": [{"text": "Final chunk of audio", "start": 0.0, "end": 3.1}, ...],
                    "duration": 20.0
                }
            ]
    """
    transcribed_chunks = []

    for chunk_path in audio_chunk_paths:
        try:
            response = call_groq(chunk_path)
        except Exception as e:
            logging.error(f"transcribe_chunks failed to call_groq for {chunk_path}: {e}")
            logging.error(traceback.format_exc())
            response = None

        transcribed_chunk = {
            "segments": response.segments,
            "duration": response.duration
        }
        transcribed_chunks.append(transcribed_chunk)
    
    return transcribed_chunks

def clump_response(
    segments: List[dict], 
    added_duration: float=0.0
) -> List[dict]:
    """
        Clumps together transcription segments into longer segments based on punctuation and duration.

        This function takes a list of transcription segments and combines them into longer segments based on the following rules:
        1. If a segment ends with a punctuation mark (period, question mark, or exclamation point), it is considered a complete segment and added to the formatted_segments list.
        2. If a segment does not end with punctuation but the total duration of the current clumped segment exceeds 60 seconds, each individual segment is added to the formatted_segments list as is.
        3. If a segment does not end with punctuation and the total duration is less than 60 seconds, it is combined with the next segment.
        4. The start_time and end_time of each formatted segment are adjusted by adding the added_duration parameter to account for the duration of previous audio chunks.

        Args:
            segments (List[dict]): A list of dictionaries representing the transcription segments. Each dictionary should have the following keys:
                - "start": The start time of the segment in seconds.
                - "end": The end time of the segment in seconds. 
                - "text": The transcription text of the segment.
            added_duration (float): The duration in seconds to add to the start and end times of each segment. This is used when combining segments from multiple audio chunks.

        Returns:
            List[dict]: A list of dictionaries representing the clumped segments. Each dictionary has the following keys:
                - "start_time": The adjusted start time of the segment in seconds.
                - "end_time": The adjusted end time of the segment in seconds.
                - "text": The transcription text of the clumped segment.

        Example:
            >>> segments = [
                    {"start": 0.0, "end": 2.5, "text": "This is the first segment"},
                    {"start": 2.5, "end": 5.0, "text": " and this is the second part."},
                    {"start": 5.0, "end": 7.0, "text": "A new sentence starts here"},
                    {"start": 7.0, "end": 10.0, "text": " but it doesn't end with punctuation"},
                    {"start": 10.0, "end": 12.5, "text": " so it continues until this part!"}
                ]
            >>> clumped_segments = clump_response(segments, added_duration=30.0)
            >>> print(clumped_segments)
            [
                {
                    "start_time": 30.0, 
                    "end_time": 35.0,
                    "text": "This is the first segment and this is the second part."
                },
                {
                    "start_time": 35.0,
                    "end_time": 42.5, 
                    "text": "A new sentence starts here but it doesn't end with punctuation so it continues until this part!"
                },  
            ]
    """
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

def default_response(
    segments: List[dict], 
    added_duration: float=0.0
) -> List[dict]:
    """
        This function takes a list of segments and an optional added_duration, and returns a list of formatted segments.

        The default_response function simply formats each segment by adding the added_duration to the start and end times, and returns a list of dictionaries containing the formatted segments.

        Args:
            segments (List[dict]): A list of dictionaries representing the segments. Each dictionary should have the following keys:
                - "start": The start time of the segment in seconds.
                - "end": The end time of the segment in seconds.
                - "text": The text content of the segment.
            added_duration (float, optional): The duration to be added to the start and end times of each segment. Defaults to 0.0.

        Returns:
            List[dict]: A list of dictionaries representing the formatted segments. Each dictionary has the following keys:
                - "start_time": The adjusted start time of the segment in seconds.
                - "end_time": The adjusted end time of the segment in seconds.
                - "text": The text content of the segment.

        Example:
            >>> segments = [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "This is the first segment."
                    },
                    {
                        "start": 5.0,
                        "end": 10.0,
                        "text": "This is the second segment."
                    }
                ]
            >>> formatted_segments = default_response(segments, added_duration=30.0)
            >>> print(formatted_segments)
            [
                {
                    "start_time": 30.0,
                    "end_time": 35.0,
                    "text": "This is the first segment."
                },
                {
                    "start_time": 35.0,
                    "end_time": 40.0,
                    "text": "This is the second segment."
                }
            ]
    """
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

def format_chunks(
    transcribed_chunks: List[dict], 
    response_type: str="default"
) -> List[dict]:
    """
        Format the transcribed chunks based on the specified response type.

        This function takes the transcribed chunks and formats them based on the specified response type. It supports two response types:
        - "clump": Combines the segments within each chunk into a single segment.
        - "default" (or any other value): Keeps the segments within each chunk separate and adjusts their start and end times based on the duration of the previous chunks.

        Args:
            transcribed_chunks (List[dict]): A list of dictionaries representing the transcribed chunks. Each dictionary should have the following keys:
                - "segments": A list of dictionaries representing the segments within the chunk. Each segment dictionary should have the keys "start", "end", and "text".
                - "duration": The duration of the chunk in seconds.
            response_type (str): The type of response to generate. Can be "clump" or "default". Defaults to "default".

        Returns:
            List[dict]: A list of dictionaries representing the formatted segments. Each dictionary has the following keys:
                - "start_time": The start time of the segment in seconds.
                - "end_time": The end time of the segment in seconds.
                - "text": The text content of the segment.

        Example:
            transcribed_chunks = [
                {
                    "segments": [
                        {"start": 0.0, "end": 2.5, "text": "This is the first segment of chunk 1."},
                        {"start": 2.5, "end": 5.0, "text": "This is the second segment of chunk 1."}
                    ],
                    "duration": 5.0
                },
                {
                    "segments": [
                        {"start": 0.0, "end": 3.0, "text": "This is the first segment of chunk 2."},
                        {"start": 3.0, "end": 6.0, "text": "This is the second segment of chunk 2."}
                    ],
                    "duration": 6.0
                }
            ]
            >>> formatted_segments = format_chunks(transcribed_chunks, response_type="clump")
            >>> print(formatted_segments)
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "This is the first segment of chunk 1. This is the second segment of chunk 1."
                },
                {
                    "start_time": 5.0,
                    "end_time": 11.0,
                    "text": "This is the first segment of chunk 2. This is the second segment of chunk 2."
                }
            ]
    """
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

def main_transcribe_audio(
    audio_chunk_paths: List[str], 
    response_type: str="clump"
) -> List[dict]:
    """
        Main function to transcribe audio chunks and format the transcription results.

        This function takes a list of file paths to audio chunks and a response type (default is "clump"). It performs the following steps:
        1. Transcribes the audio chunks using the `transcribe_chunks` function.
        2. Formats the transcribed chunks based on the specified response type using the `format_chunks` function.
        3. Returns the formatted transcription segments.

        Args:
            audio_chunk_paths (List[str]): A list of file paths to the audio chunks to be transcribed.
            response_type (str, optional): The type of response formatting to apply. Default is "clump".

        Returns:
            List[dict]: A list of dictionaries representing the formatted transcription segments. The format of the segments depends on the `response_type` argument.

        Example:
        >>> audio_chunk_paths = ["01_chunk.mp3", "02_chunk.mp3"]
        >>> response_type = "clump"
        >>> transcription_segments = main_transcribe_audio(audio_chunk_paths, response_type)
        >>> print(transcription_segments)
        [
            {
                "start_time": 0.0,
                "end_time": 20.0,
                "text": "This is the transcription for chunk 1. It contains the text from all the segments in chunk 1."
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "text": "This is part 1 of chunk 2."
            },
            {
                "start_time": 30.0,
                "end_time": 40.0,
                "text": "This is part 2 of chunk 2."
            }
        ]
    """
    transcribed_chunks = transcribe_chunks(audio_chunk_paths, "temp")
    segments = format_chunks(transcribed_chunks, response_type=response_type)
    
    return segments