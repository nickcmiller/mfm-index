from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import torch
from typing import List
import json
import os
import logging
from transcribe_video import groq_transcribe_audio, format_response

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

def merge_diarization_results_and_transcription(diarization_results: List[dict], audio_file_path: str) -> List[dict]:
    
    segments = []
    try:
        transcription_response = groq_transcribe_audio(audio_file_path)
        if transcription_response is None:
            raise ValueError("Failed to transcribe audio. The transcription response is None.")
        segments = format_response(transcription_response)
    except Exception as e:
        logging.error(f"Error during transcription or formatting: {e}")
        raise

    merged_results = []
    for d in diarization_results:
        combined_text = ""
        speaker = d["speaker"]
        used_segments = []
        for segment in segments:
            if segment["start_time"] < d["end_time"]-1:
                combined_text += segment["text"]
                used_segments.append(segment)
        segments = [segment for segment in segments if segment not in used_segments]
        d["transcription"] = combined_text
        merged_results.append(d)

    return merged_results

def create_transcript(transcribed_segments: list) -> str:
    transcript = ""
    for segment in transcribed_segments:
        transcript += f"{segment['speaker']}: {segment['transcription']}\n\n"
    return transcript

def create_diarized_transcript(audio_file_path: str) -> str:
    diarization_results = diarize_audio(audio_file_path)
    transcription = format_response(groq_transcribe_audio(audio_file_path))
    merged_results = merge_diarization_results_and_transcription(diarization_results, audio_file_path)
    return create_transcript(merged_results)

# if __name__ == "__main__":
