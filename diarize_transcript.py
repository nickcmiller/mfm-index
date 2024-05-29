from pyannote.audio import Pipeline
import torch

import logging
import time
from dotenv import load_dotenv

from transcribe_audio import create_audio_chunks, transcribe_chunks
from pydub import AudioSegment
from typing import List, Dict
import json
import os



logging.basicConfig(level=logging.DEBUG)

load_dotenv()
# Need to accept the pyannote terms of service on Hugging Face website
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def diarize_audio(audio_file_path: str) -> List[dict]:
    """
    This function diarizes an audio file using the pyannote library.

    Args:
        audio_file_path (str): The path to the audio file to be diarized.

    Returns:
        list: A list of dictionaries containing the start time, end time, and speaker label for each segment of the audio file.
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

def merge_diarization_results_and_transcription(diarization_results: List[dict], segments: List[dict]) -> List[dict]:
    merged_results = []

    def overlap_percentage(segment, diarization):
        start = max(segment["start_time"], diarization["start_time"])
        end = min(segment["end_time"], diarization["end_time"])
        overlap = max(0, end - start)
        segment_duration = segment["end_time"] - segment["start_time"]
        return (overlap / segment_duration) * 100

    for d in diarization_results:
        combined_text = ""
        speaker = d["speaker"]
        used_segments = []
        for segment in segments:

            overlap = overlap_percentage(segment, d)
            if overlap > 50:  # Using 50% as the threshold for inclusion
                combined_text += f"|{segment['text']}|"
                used_segments.append(segment)
            elif segment["end_time"] < d["end_time"]:
                combined_text += f"|{segment['text']}|"
                used_segments.append(segment)
            elif segment == segments[-1] and d == diarization_results[-1]:
                combined_text += f"|{segment['text']}|"
                used_segments.append(segment)
        segments = [segment for segment in segments if segment not in used_segments]
        if len(combined_text) > 0:
            d["transcription"] = f"{str(d['start_time'])} - {str(d['end_time'])}: {combined_text}"
            logging.info(f"Transcription for {d['speaker']}: {d['transcription']}")
            merged_results.append(d)
        else:
            logging.info(f"No text found for {d['speaker']}: {d['start_time']} - {d['end_time']}")

    return merged_results

def create_transcript(transcribed_segments: list) -> str:
    transcript = ""
    for segment in transcribed_segments:
        transcript += f"{segment['speaker']}: {segment['transcription']}\n\n"
    return transcript

def create_diarized_transcript(audio_file_path: str, diarization_results: List[dict]=None, segments: List[dict]=None) -> str:

    if diarization_results is None:
        diarization_results = diarize_audio(audio_file_path)

    if segments is None:
        try:
            segments = transcribe_chunks(audio_file_path, "temp")
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise

    merged_results = merge_diarization_results_and_transcription(diarization_results, segments)

    return create_transcript(merged_results)

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
        segments = main_transcribe_audio(audio_file_path, response_type="clump")
        with open("segments.json", "w") as f:
            json.dump(segments, f, indent=4)
    else:
        with open("segments.json", "r") as f:
            segments = json.load(f)

    if new_create is True:
        from transcribe_audio import create_audio_chunks
        audio_chunks = create_audio_chunks(audio_file_path, "temp")
        diarization_results = []
        for audio_chunk in audio_chunks:
            diarization_chunk = diarize_audio(audio_chunk)
            diarization_results.extend(diarization_chunk)
        with open("diarization_results.json", "w") as f:
            json.dump(diarization_results, f, indent=4)
    else:
        with open("diarization_results.json", "r") as f:
            diarization_results = json.load(f)

    result = create_diarized_transcript(audio_file_path, diarization_results, formatted_chunks)
    with open("transcript.txt", "w") as f:
        f.write(result)
    # print(result)
