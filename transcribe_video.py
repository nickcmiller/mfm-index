from groq import Groq
import os
from typing import List, Optional
import logging
import traceback
import shutil
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def groq_transcribe_audio(audio_chunk_file: str) -> Optional[str]:
    """
    Transcribe the audio file using Groq Whisper API.

    Args:
        audio_chunk_file (str): The path to the chunk of audio to transcribe.

    Returns:
        str: The transcribed text, or None if the transcription failed.
    """
    client = Groq()

    # Transcribe the audio file
    try:
        # Get the file size in megabytes and log it
        file_size_bytes = os.path.getsize(audio_chunk_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        logging.info(f"The size of the file sent to Groq Whisper is {file_size_megabytes} MB.")

        with open(audio_chunk_file, "rb") as af:
            transcript_chunk_file = client.audio.transcriptions.create(
                file=(audio_chunk_file, af.read()),
                model="whisper-large-v3",
            )
        return transcript_chunk_file.text
    except FileNotFoundError:
        logging.error(f"File {audio_chunk_file} not found.")
        return None
    except Exception as e:
        logging.error(f"groq_transcribe_audio failed to transcribe audio file {audio_chunk_file}: {e}")
        return None

if __name__ == "__main__":
    audio_chunk_file = "Former OPD Chief LeRonne Armstrong announces city council run.mp3"
    transcribed_text = groq_transcribe_audio(audio_chunk_file)
    print(transcribed_text)

