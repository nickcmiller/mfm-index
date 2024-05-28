from groq import Groq
import os
from typing import List, Optional
import logging
import traceback
import shutil
from dotenv import load_dotenv
import time
import httpx
import json

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def groq_transcribe_audio(audio_chunk_file: str) -> str:
    """
    This function transcribes an audio chunk file using the Groq library.

    Args:
        audio_chunk_file (str): The path to the audio chunk file to be transcribed.

    Returns:
        str: JSON with the transcribed audio.
    """
    client = Groq()
    max_retries = 6
    retry_delay = 10  # seconds

    # Transcribe the audio file
    try:
        # Get the file size in megabytes and log it
        file_size_bytes = os.path.getsize(audio_chunk_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        logging.info(f"The size of the file sent to Groq Whisper is {file_size_megabytes} MB.")

        for attempt in range(max_retries):
            try:
                with open(audio_chunk_file, "rb") as af:
                    transcript_chunk_file = client.audio.transcriptions.create(
                        file=(audio_chunk_file, af.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json"
                    )
                time.sleep(3)
                return json.dumps(transcript_chunk_file.segments, indent=4)
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
        logging.error(f"File {audio_chunk_file} not found.")
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

def format_response(response: json):
    segments = []
    for segment in response:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        segments.append({
            "start_time": start_time, 
            "end_time": end_time, 
            "text": text.strip()
        })
    return segments

if __name__ == "__main__":
    response = groq_transcribe_audio('Former OPD Chief LeRonne Armstrong announces city council run.mp3')
    parsed_response = json.loads(response)
    print(format_response(parsed_response))
