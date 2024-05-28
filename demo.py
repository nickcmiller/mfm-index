from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import torch
from typing import List
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
        counter_padding = str(counter).zfill(5)
        file_path = os.path.join(speaker_segments_dir, f"{counter_padding}_{speaker_label}_{audio_file_base_name}")
        speaker_segment.export(file_path, format="mp3")
        logging.debug(f"Exported speaker segment {counter} to {file_path}")

        speaker_segment_list.append([speaker_label, file_path])
        
        # Increment counter
        counter += 1
    
    # Sort speaker_segment_dir by the integer in the filename
    speaker_segments_dir 
    
    # Remove the audio file
    os.remove(audio_file_path)

    return speaker_segment_list

def combine_speaker_segments(speaker_segments: list,divider_path: str, max_file_size: int) -> list:
    try:
        # Continue with the rest of the function as before
        divider = AudioSegment.from_file(divider_path)
        speaker_segments_dir = os.path.join(os.getcwd(), "speaker_segments")
        mb_max = max_file_size

        combined_paths=[]
        speaker_list = [s[0] for s in speaker_segments]
        segments_paths = [s[1] for s in speaker_segments]
        counter = 1

        current_audio = AudioSegment.empty()
        for index, s in enumerate(segments_paths):
            new_audio = AudioSegment.from_file(s)
            combined_audio = current_audio + divider + new_audio
            estimated_size_mb = ((len(combined_audio.raw_data)) / (1024 * 1024)) * (1 / 12)

            logging.info(f"Processing segment {index + 1}/{len(speaker_segments)}")
            logging.info(f"Max audio size: {mb_max} MB")
            logging.info(f"Estimated audio size: {estimated_size_mb} MB") 
            logging.info(f"Size difference: {mb_max - estimated_size_mb} MB")

            if estimated_size_mb > mb_max or index == len(segments_paths) - 1:
                if estimated_size_mb > mb_max:
                    logging.debug("Current combined audio exceeds max file size, exporting...")
                    counter += 1
                    current_audio = new_audio
                else:
                    logging.debug("Last segment reached, exporting...")

                file_path = os.path.join(speaker_segments_dir, f"{counter}_combined_segment.mp3")
                combined_audio.export(file_path, format="mp3")
                combined_paths.append(file_path)
            else:
                current_audio = combined_audio

        logging.info(f"Combined {len(speaker_segments)} segments into {len(combined_paths)} files.")
        logging.info(f"Combined paths: {combined_paths}")

        return {
            "speaker_list": speaker_list,
            "combined_paths": combined_paths
        }
    except FileNotFoundError as e:
        logging.error(f"An error occurred while combining speaker segments: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while combining speaker segments: {e}")
        raise

def transcribe_combined_segments(combined_segments: list, speaker_list: list) -> list:
    transcribed_segments = []
    for file_path in combined_segments:
        try:
            transcribed_text = groq_transcribe_audio(file_path)
            if transcribed_text is not None:
                transcribed_segments.append(transcribed_text)
            else:
                logging.error(f"Failed to transcribe combined segment: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Failed to remove combined segment: {file_path}. Error: {e}")
        except FileNotFoundError:
            logging.error(f"Audio file not found: {file_path}")
        except PermissionError:
            logging.error(f"Permission denied for accessing the file: {file_path}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while transcribing {file_path}: {e}")
    
    if os.path.exists("speaker_segments"):
        shutil.rmtree("speaker_segments")

    combined_list = []
    for segment in transcribed_segments:
        segment_list = [s.strip() for s in segment.split("Divider, divider, divider.") if s.strip() != ""]
        print(f"Segment List: {segment_list}")
        combined_list.extend(segment_list)

    speaker_segment_list = []
    for segment in combined_list:
        print(f"len(speaker_list): {len(speaker_list)}")
        print(f"len(combined_list): {len(combined_list)}")
        speaker_segment_list.append({
            "speaker": speaker_list[combined_list.index(segment)],
            "text": segment
        })
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
        except Exception as e:
            logging.error(f"An unexpected error occurred while transcribing {file_path}: {e}")

    if os.path.exists("speaker_segments"):
        shutil.rmtree("speaker_segments")

    return transcribed_segments

def create_transcript(transcribed_segments: list) -> str:
    transcript = ""
    for segment in transcribed_segments:
        transcript += f"{segment['speaker']}: {segment['text']}\n\n"
    return transcript


if __name__ == "__main__":
    youtube_url="https://www.youtube.com/watch?v=XJvQ97eMOFU&ab_channel=talkSPORT"
    audio_file_path = yt_dlp_download(youtube_url)
    # diarization_results = diarize_audio(audio_file_path)
    # with open("diarization_results.txt", "w") as f:
    #     f.write(str(diarization_results))
    # for result in diarization_results:
    #     print(result)
    diarization_results_str = open("diarization_results.txt", "r").read()
    diarization_results = eval(diarization_results_str)
    segments = save_speaker_segments(diarization_results, audio_file_path)
    combined_segments = combine_speaker_segments(segments, "divider.mp3", 23)
    transcribed_segments = transcribe_combined_segments(combined_segments["combined_paths"], combined_segments["speaker_list"])
    for segment in transcribed_segments:
        print(segment)
    transcript = create_transcript(transcribed_segments)
    with open("transcript.txt", "w") as f:
        f.write(transcript)

