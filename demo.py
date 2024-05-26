from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import torch
import os
import logging
import shutil
import time
from download_video import yt_dlp_download
from transcribe_video import groq_transcribe_audio

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
# Need to accept the pyannote terms of service on Hugging Face website
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def diarize_audio(audio_file_path: str) -> list:
    """
    This function diarizes an audio file using the pyannote library.

    Args:
        audio_file_path (str): The path to the audio file to be diarized.

    Returns:
        list: A list of tuples containing the start time, end time, and speaker label for each segment of the audio file.
    """
    # Record the start time
    start_time = time.time()  

    try:
        logging.debug("Loading the pretrained pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_ACCESS_TOKEN
        )
        logging.debug("Pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the pipeline: {e}")
        raise

    # Check if pipeline is successfully created
    if pipeline is None:
        raise ValueError("Failed to load the pipeline. Check your Hugging Face access token.")

    # Send pipeline to GPU if CUDA is available
    if torch.cuda.is_available():
        try:
            logging.debug("Sending pipeline to GPU...")
            pipeline.to(torch.device("cuda"))
            logging.debug("Pipeline sent to GPU successfully.")
        except Exception as e:
            logging.error(f"Failed to send pipeline to GPU: {e}")
            raise

    # Apply pretrained pipeline
    try:
        logging.debug(f"Applying the pipeline to the audio file: {audio_file_path}")
        diarization = pipeline(audio_file_path)
        logging.debug("Pipeline applied successfully.")
    except Exception as e:
        logging.error(f"Failed to apply pretrained pipeline: {e}")
        raise

    # Process the diarization results
    diarization_results = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    
    # Record the end time and log the elapsed time
    end_time = time.time()  
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")

    return diarization_results

def save_speaker_segments(diarization_results: list, audio_file_path: str) -> list:
    # Load your audio file
    audio_file = AudioSegment.from_file(audio_file_path)
    
    # Get the base name of the audio file
    audio_file_base_name = os.path.basename(audio_file_path)

    # Create speaker_segments directory in the current directory and set file path to it
    if not os.path.exists("speaker_segments"):
        os.makedirs("speaker_segments")
    speaker_segments_dir = os.path.join(os.getcwd(), "speaker_segments")

    # Initialize speaker_segments list
    speaker_segment_list = []

    # Initialize counter
    counter = 1

    # Process each diarization result
    for start_time, end_time, speaker_label in diarization_results:
        start_time_str = str(start_time)
        end_time_str = str(end_time)
        
        start_ms = int(float(start_time_str[:-1]) * 1000)  # Convert start time to milliseconds
        end_ms = int(float(end_time_str[:-1]) * 1000)  # Convert end time to milliseconds
        
        # Extract the segment corresponding to the speaker
        speaker_segment = audio_file[start_ms:end_ms]
        
        # Save the speaker segment to the speaker_segments directory
        file_path = os.path.join(speaker_segments_dir, f"{counter}_{speaker_label}_{audio_file_base_name}")
        speaker_segment.export(file_path, format="mp3")
        logging.debug(f"Exported speaker segment {counter} to {file_path}")

        speaker_segment_list.append([speaker_label, file_path])
        
        # Increment counter
        counter += 1
    
    # Remove the audio file
    os.remove(audio_file_path)

    return speaker_segment_list

def transcribe_speaker_segments(speaker_segments: list) -> list:
    transcribed_segments = []
    for speaker_label, file_path in speaker_segments:
        try:

            transcribed_text = groq_transcribe_audio(file_path)

            if transcribed_text is not None:
                transcribed_segments.append({
                    "speaker": speaker_label,
                    "text": transcribed_text
                })
            else:
                logging.error(f"Failed to transcribe speaker segment: {file_path}")

            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Failed to remove speaker segment: {file_path}. Error: {e}")

        except FileNotFoundError:
            logging.error(f"Audio file not found: {file_path}")
        except PermissionError:
            logging.error(f"Permission denied for accessing the file: {file_path}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logging.error(f"Too many requests: unable to transcribe due to rate limits on {file_path}")
            else:
                logging.error(f"HTTP error occurred while transcribing {file_path}: Status code {e.response.status_code}")
        except httpx.RequestError as e:
            logging.error(f"Request error occurred while transcribing {file_path}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while transcribing {file_path}: {e}")
        
        time.sleep(3)

    if os.path.exists("speaker_segments"):
        shutil.rmtree("speaker_segments")

    return transcribed_segments

def create_transcript(transcribed_segments: list) -> str:
    transcript = ""
    for segment in transcribed_segments:
        transcript += f"{segment['speaker']}: {segment['text']}\n\n"
    return transcript


if __name__ == "__main__":
    youtube_url="https://www.youtube.com/watch?v=miD5NsLnCMg&ab_channel=KTVUFOX2SanFrancisco"
    audio_file_path = yt_dlp_download(youtube_url)
    diarization_results = diarize_audio(audio_file_path)
    with open("diarization_results.txt", "w") as f:
        f.write(str(diarization_results))
    for result in diarization_results:
        print(result)
    # diarization_results_str = open("diarization_results.txt", "r").read()
    # diarization_results = eval(diarization_results_str)
    segments = save_speaker_segments(diarization_results, audio_file_path)
    transcribed_segments = transcribe_speaker_segments(segments)
    for segment in transcribed_segments:
        print(segment)
    transcript = create_transcript(transcribed_segments)
    with open("transcript.txt", "w") as f:
        f.write(transcript)

