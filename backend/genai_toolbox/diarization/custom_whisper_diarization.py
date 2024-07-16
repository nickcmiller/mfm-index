from pyannote.audio import Pipeline
import torch

import logging
import time
import os

from pydub import AudioSegment
from typing import List, Dict
import json


logging.basicConfig(level=logging.DEBUG)

# Need to accept the pyannote terms of service on Hugging Face website
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def diarize_audio(audio_file_path: str) -> List[dict]:
    """
    Diarize an audio file using a pretrained speaker diarization pipeline.

    This function takes an audio file path as input and applies a pretrained speaker diarization pipeline to identify and segment the audio based on different speakers. It uses the pyannote.audio library and the "pyannote/speaker-diarization-3.1" pretrained model from Hugging Face.

    Args:
        audio_file_path (str): The path to the audio file to be diarized.

    Returns:
        List[dict]: A list of dictionaries representing the diarized segments. Each dictionary contains the following keys:
            - "start_time" (float): The start time of the segment in seconds.
            - "end_time" (float): The end time of the segment in seconds.
            - "speaker" (str): The identified speaker label for the segment.

    Raises:
        ValueError: If the pipeline fails to load due to an invalid Hugging Face access token.
        Exception: If there is an error loading the pipeline, sending it to the GPU (if available), or applying it to the audio file.

    Example:
        >>> audio_file_path = "path/to/audio/file.wav"
        >>> diarization_results = diarize_audio(audio_file_path)
        >>> print(diarization_results)
        [
            {"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_00"},
            {"start_time": 2.5, "end_time": 5.0, "speaker": "SPEAKER_01"},
            ...
        ]
    """
    # Record the start time
    start_time = time.time()  

    try:
        logging.info("Loading the pretrained pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_ACCESS_TOKEN
        )
        logging.info("Pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the pipeline: {e}")
        raise

    # Check if pipeline is successfully created
    if pipeline is None:
        raise ValueError("Failed to load the pipeline. Check your Hugging Face access token.")

    # Send pipeline to GPU if CUDA is available
    if torch.cuda.is_available():
        try:
            logging.info("Sending pipeline to GPU...")
            pipeline.to(torch.device("cuda"))
            logging.info("Pipeline sent to GPU successfully.")
        except Exception as e:
            logging.error(f"Failed to send pipeline to GPU: {e}")
            raise

    # Apply pretrained pipeline
    try:
        logging.info(f"Applying the pipeline to the audio file: {audio_file_path}")
        diarization = pipeline(audio_file_path)
        logging.info("Pipeline applied successfully.")
    except Exception as e:
        logging.error(f"Failed to apply pretrained pipeline: {e}")
        raise

    # Process the diarization results
    diarization_results = [{"start_time": turn.start, "end_time": turn.end, "speaker": speaker} for turn, _, speaker in diarization.itertracks(yield_label=True)]
    
    # Record the end time and log the elapsed time
    end_time = time.time()  
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")

    return diarization_results

def diarize_audio_chunks(audio_file_paths: List[str]) -> List[dict]:
    """
    Diarizes a list of audio chunks and returns the combined diarization results.

    This function takes a list of file paths to audio chunks and performs speaker diarization on each chunk using the `diarize_audio` function. The diarization results from each chunk are then combined into a single list, with the start and end times adjusted based on the duration of the previous chunks.

    Args:
        audio_file_paths (List[str]): A list of file paths to the audio chunks to be diarized.

    Returns:
        List[dict]: A list of dictionaries representing the combined diarization results. Each dictionary contains the following keys:
            - "start_time" (float): The start time of the speaker segment in seconds, adjusted based on the duration of previous chunks.
            - "end_time" (float): The end time of the speaker segment in seconds, adjusted based on the duration of previous chunks.
            - "speaker" (str): The identified speaker label for the segment.

    Example:
        >>> audio_file_paths = ["01_chunk.mp3", "02_chunk.mp3", "03_chunk.mp3"]
        >>> diarization_results = diarize_audio_chunks(audio_file_paths)
        >>> print(diarization_results)
        [
            {"start_time": 0.0, "end_time": 5.2, "speaker": "SPEAKER_00"},
            {"start_time": 5.2, "end_time": 12.8, "speaker": "SPEAKER_01"},
            {"start_time": 12.8, "end_time": 18.5, "speaker": "SPEAKER_00"},
            ...
        ]
    """
    combined_diarization_results = []
    added_duration = 0
    for path in audio_file_paths:
        diarization_chunk_results = diarize_audio(path)
        for result in diarization_chunk_results:
            result["start_time"] += added_duration
            result["end_time"] += added_duration
            combined_diarization_results.append(result)
        added_duration = combined_diarization_results[-1]["end_time"]
    
    return combined_diarization_results
    
def condense_diarization_results(diarized_segments: List[dict]) -> List[dict]:
    """
    This function takes a list of diarized segments and condenses them by combining consecutive segments from the same speaker.

    The process is as follows:
    1. Remove segments that are less than 1 second long.
    2. Initialize variables for combining consecutive segments:
        - `condensed_results`: an empty list to store the condensed segments
        - `current_segment`: a variable to keep track of the current segment being processed
        - `tolerance`: the maximum allowed gap (in seconds) between consecutive segments to be combined
    3. Iterate through the diarized segments:
        - If `current_segment` is None, set it to the current segment.
        - If the current segment's speaker is the same as the previous segment's speaker and the gap between them is less than or equal to the `tolerance`, update the end time of the `current_segment` to the end time of the current segment.
        - If the current segment's speaker is different or the gap is greater than the `tolerance`, append the `current_segment` to the `condensed_results` and set the `current_segment` to the current segment.
    4. After the loop, if the `current_segment` is not None and hasn't been added to the `condensed_results`, append it.
    5. Return the `condensed_results`.

    Args:
        diarized_segments (List[dict]): A list of dictionaries representing the diarized segments. Each dictionary should have the following keys:
            - "start_time": The start time of the segment in seconds.
            - "end_time": The end time of the segment in seconds.
            - "speaker": The identified speaker for the segment.

    Returns:
        List[dict]: A list of dictionaries representing the condensed diarized segments. Each dictionary has the same keys as the input segments.

    Example:
        >>> diarized_segments = [    
            {"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_00"},
            {"start_time": 2.5, "end_time": 5.0, "speaker": "SPEAKER_00"},
            ...
        ]
        >>> condensed_results = condense_diarization_result(diarized_segments)
        >>> print(condensed_results)
        [
            {"start_time": 0.0, "end_time": 5.0, "speaker": "SPEAKER_00"},
            ...
        ]   
    """
    # Remove segments that are less than a second long
    diarized_segments = [segment for segment in diarized_segments if segment['end_time'] - segment['start_time'] >= 1.0]

    # Initialize variables for combining consecutive segments
    condensed_results = []
    current_segment = None
    tolerance = 3

    # Iterate through the data to combine consecutive segments
    for segment in diarized_segments:
        if current_segment is None:
            current_segment = segment
        elif current_segment["speaker"] == segment["speaker"] and abs(current_segment["end_time"] - segment["start_time"]) <= tolerance:
            current_segment["end_time"] = segment["end_time"]
        else:
            condensed_results.append(current_segment)
            current_segment = segment
    
    # Append the last segment if it hasn't been added to the condensed results
    if current_segment is not None and current_segment not in condensed_results:
        condensed_results.append(current_segment)

    return condensed_results

def diarize_and_condense_audio_chunks(audio_file_paths: List[str]) -> List[dict]:
    """
    Diarizes and condenses the audio chunks.

    This function takes a list of file paths to audio chunks and performs the following steps:
    1. Diarizes the audio chunks using the `diarize_audio_chunks` function. This function uses a speaker diarization model (PyAnnote) to identify and segment the audio based on different speakers.
    2. Condenses the diarized segments using the `condense_diarization_results` function. This function combines consecutive segments from the same speaker to create a more condensed representation of the diarization results.
    3. Returns the condensed diarization segments.

    Args:
        audio_file_paths (List[str]): A list of file paths to the audio chunks.

    Returns:
        List[dict]: A list of dictionaries representing the condensed diarized segments. Each dictionary likely contains information such as the start time, end time, and speaker for each segment.

    Example:
    >>> audio_file_paths = ["01_chunk.mp3", "02_chunk.mp3", "03_chunk.mp3"]
    >>> diarized_and_condensed_segments = diarize_and_condense_audio_chunks(audio_file_paths)
    >>> print(diarized_and_condensed_segments)
    [
        {"start_time": 0.0, "end_time": 5.0, "speaker": "SPEAKER_00"},
        {"start_time": 5.0, "end_time": 12.5, "speaker": "SPEAKER_01"},
        {"start_time": 12.5, "end_time": 20.0, "speaker": "SPEAKER_00"}
    ]
    """
    diarized_segments = diarize_audio_chunks(audio_file_paths)
    condensed_segments = condense_diarization_results(diarized_segments)
    return condensed_segments