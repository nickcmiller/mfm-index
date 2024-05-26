from groq import Groq
import os
from typing import List, Optional
import logging
import traceback
import shutil
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def groq_transcribe_audio(audio_chunk_file: str) -> str:
    """
    This function transcribes an audio chunk file using the Groq library.

    Args:
        audio_chunk_file (str): The path to the audio chunk file to be transcribed.

    Returns:
        str: The transcribed text
    """
    client = Groq()
    max_retries = 5
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
                    )
                return transcript_chunk_file.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logging.info(f"Received 429 response, waiting {retry_delay} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"HTTP error occurred: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error occurred: {e}")
                raise
        logging.error("Max retries reached, failed to transcribe audio.")
        return None
    except FileNotFoundError:
        logging.error(f"File {audio_chunk_file} not found.")
        raise
    except Exception as e:
        logging.error(f"groq_transcribe_audio failed to transcribe audio file {audio_chunk_file}: {e}")
        raise

