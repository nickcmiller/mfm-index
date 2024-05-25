from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import torch
import os
import logging
import time
from download_video import yt_dlp_download
from transcribe_video import groq_transcribe_audio

logging.basicConfig(level=logging.INFO)

load_dotenv()
# Need to accept the pyannote terms of service on Hugging Face website
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def diarize_audio(audio_file_path: str) -> list:
    # Record the start time
    start_time = time.time()  

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_ACCESS_TOKEN)

    # Check if pipeline is successfully created
    if pipeline is None:
        raise ValueError("Failed to load the pipeline. Check your Hugging Face access token.")

    # send pipeline to GPU if CUDA is available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Apply pretrained pipeline
    diarization = pipeline("Former OPD Chief LeRonne Armstrong announces city council run.mp3")

    # Assuming diarization_results is a list of tuples with turn.start and turn.end as floats
    diarization_results = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    
    # Record the end time and log the elapsed time
    end_time = time.time()  
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")

    return diarization_results

def save_speaker_segments(diarization_results, audio_file_path):
    # Load your audio file
    audio_file = AudioSegment.from_file(audio_file_path)
    
    # Get the base name of the audio file
    audio_file_base_name = os.path.basename(audio_file_path)

    # Create speaker_segments directory in the current directory and set file path to it
    if not os.path.exists("speaker_segments"):
        os.makedirs("speaker_segments")
    speaker_segments_dir = os.path.join(os.getcwd(), "speaker_segments")

    # Initialize speaker_segments list
    speaker_segments = []

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

        speaker_segments.append([speaker_label, file_path])
        
        # Increment counter
        counter += 1

    return speaker_segments

def transcribe_speaker_segments(speaker_segments):
    transcribed_segments = []
    for speaker_label, file_path in speaker_segments:
        transcribed_text = groq_transcribe_audio(file_path)
        transcribed_segments.append({
            "speaker": speaker_label,
            "text": transcribed_text
        })
    return transcribed_segments

def create_transcript(transcribed_segments):
    transcript = ""
    for segment in transcribed_segments:
        transcript += f"{segment['speaker']}: {segment['text']}\n\n"
    return transcript


if __name__ == "__main__":
    youtube_url="https://www.youtube.com/watch?v=PdFedYcfWmE&ab_channel=BeanymanSports"
    audio_file_path = yt_dlp_download(youtube_url)
    diarization_results = diarize_audio(audio_file_path)
    print(diarization_results)
    segments = save_speaker_segments(diarization_results, audio_file_path)
    print(segments)
    transcribed_segments = transcribe_speaker_segments(segments)
    print(transcribed_segments)
    transcript = create_transcript(transcribed_segments)
    with open("transcript.txt", "w") as f:
        f.write(transcript)

