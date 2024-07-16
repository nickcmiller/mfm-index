import os

from typing import List
from pydub import AudioSegment
import json
import logging
import traceback
import shutil

from genai_toolbox.download_sources.download_video import yt_dlp_download
from genai_toolbox.diarization.custom_whisper_diarization import diarize_and_condense_audio_chunks
from genai_toolbox.transcription.whisper_transcription_functions import main_transcribe_audio

def create_audio_chunks(audio_file_path: str, temp_dir: str, chunk_size: int=25*60000) -> List[str]:
    """
    Splits an audio file into smaller segments or chunks based on a specified duration. This function is useful for processing large audio files incrementally or in parallel, which can be beneficial for tasks such as audio analysis or transcription where handling smaller segments might be more manageable.

    AudioSegment can slice an audio file by specifying the start and end times in milliseconds. This allows you to extract precise segments of the audio without needing to process the entire file at once. For example, `audio[1000:2000]` extracts a segment from the 1-second mark to the 2-second mark of the audio file.

    Args:
        audio_file (str): The absolute or relative path to the audio file that needs to be chunked. This file should be accessible and readable.
        
        chunk_size (int): The length of each audio chunk expressed in milliseconds. This value determines how the audio file will be divided. For example, a `chunk_size` of 1000 milliseconds will split the audio into chunks of 1 second each.
        
        temp_dir (str): The directory where the temporary audio chunk files will be stored. This directory will be used to save the output chunk files, and it must have write permissions. If the directory does not exist, it will be created.

    Returns:
        List[str]: A list containing the file paths of all the audio chunks created. Each path in the list represents a single chunk file stored in the specified `temp_dir`. The files are named sequentially based on their order in the original audio file.

    Raises:
        FileNotFoundError: If the `audio_file` does not exist or is inaccessible.
        
        PermissionError: If the script lacks the necessary permissions to read the `audio_file` or write to the `temp_dir`.
        ValueError: If `chunk_size` is set to a non-positive value.

    Example: 
        >>> audio_file_path = "path/to/audio/file.mp3"
        >>> temp_dir = "path/to/temp/directory"
        >>> chunk_size = 30000  # 30 seconds
        >>> audio_chunk_paths = create_audio_chunks(audio_file_path, temp_dir, chunk_size)
        >>> print(audio_chunk_paths)
        [
            "path/to/temp/directory/0_chunk.mp3",
            "path/to/temp/directory/1_chunk.mp3",
            "path/to/temp/directory/2_chunk.mp3",
            ...
        ]
    """
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as e:
        logging.error(f"create_audio_chunks failed to load audio file {audio_file_path}: {e}")
        logging.error(traceback.format_exc())
        return []

    start = 0
    end = chunk_size
    counter = 0
    audio_chunk_paths = []

    while start < len(audio):
        chunk = audio[start:end]
        chunk_file_path = os.path.join(temp_dir, f"{counter}_{file_name}.mp3")
        try:
            chunk.export(chunk_file_path, format="mp3") # Using .mp3 because it's cheaper
            audio_chunk_paths.append(chunk_file_path)
        except Exception as e:
            error_message = f"create_audio_chunks failed to export chunk {counter}: {e}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            raise error_message
        start += chunk_size
        end += chunk_size
        counter += 1
    return audio_chunk_paths

def merge_diarization_and_transcription(diarization_results: List[dict], transcription_results: List[dict]) -> List[dict]:
    """
    Merge diarization and transcription results based on overlap percentage between segments.

    1. Initialization: 
        - An empty list merged_results is created to store the results after merging.
    2. Overlap Calculation:
        - The function defines a nested function overlap_percentage to calculate the percentage of overlap between a transcription segment and a diarization segment. It calculates the overlap duration and then divides it by the total duration of the transcription segment to get the percentage.
    3. Merging Logic:
        - The function iterates over each diarization result.
        - For each diarization segment, it initializes an empty string combined_text to accumulate the text from overlapping transcription segments.
        - It then iterates over each transcription segment to check if it overlaps significantly by the defined amount with the diarization segment.
        - If a defined overlap is found, the text from the transcription segment is appended to combined_text.
        - Segments that are used are tracked and removed from further consideration to prevent duplication.
    4. Finalizing Merged Segment:
        - After processing all relevant transcription segments for a diarization segment, if any text has been combined, it is formatted and added to the diarization segment under the key transcription.
        - The updated diarization segment (now containing transcription data) is added to merged_results.
    5. Logging:
        - Throughout the process, various logging statements provide debug information about the merging process, such as when text is found for a speaker or when no text is found.
    6. Outcome:
        - The function effectively combines the speaker identification data from diarization with the corresponding transcribed text based on temporal overlap, resulting in a richer dataset that includes both who was speaking and what was said during each segment.


    Args:
        diarization_results (List[dict]): A list of dictionaries representing the diarization segments. Each dictionary should have the following keys:
            - "start_time": The start time of the segment in seconds.
            - "end_time": The end time of the segment in seconds.
            - "speaker": The identified speaker for the segment.
        segments (List[dict]): A list of dictionaries representing the transcription segments. Each dictionary should have the following keys:
            - "start_time": The start time of the segment in seconds.
            - "end_time": The end time of the segment in seconds. 
            - "text": The transcribed text for the segment.

    Returns:
        List[dict]: A list of dictionaries representing the merged diarization and transcription results. Each dictionary has the following keys:
            - "start_time": The start time of the segment in seconds.
            - "end_time": The end time of the segment in seconds.
            - "speaker": The identified speaker for the segment.
            - "transcription": The transcribed text for the segment.

    Example:
        diarization_results = [
            {"start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_00"},
            {"start_time": 2.5, "end_time": 5.0, "speaker": "SPEAKER_01"},
            {"start_time": 5.0, "end_time": 7.5, "speaker": "SPEAKER_00"},
            ...
        ]
        transcription_results = [
            {"start_time": 0.0, "end_time": 1.5, "text": "Hey, how have you been?"},
            {"start_time": 1.5, "end_time": 4.0, "text": "I've been doing very well, thank you for asking"},
            {"start_time": 4.0, "end_time": 6.5, "text": "That's great to hear. I'm ready to discuss the project proposal."},
            ...
        ]

        >>> merged_results = merge_diarization_and_transcription(diarization_results, transcription_results)
        >>> print(merged_results)

        [
            {
                "start_time": 0.0,
                "end_time": 2.5,
                "speaker": "SPEAKER_00",
                "transcription": "Hey, how have you been?"
            },
            {
                "start_time": 2.5,
                "end_time": 5.0,
                "speaker": "SPEAKER_01",
                "transcription": "I've been doing very well, thank you for asking"
            },
            {
                "start_time": 5.0,
                "end_time": 7.5,
                "speaker": "SPEAKER_00",
                "transcription": "That's great to hear. I'm ready to discuss the project proposal."
            },
            ...
        ]

    """
    def overlap_percentage(transcription: dict, diarization: dict) -> float:
        """
        Calculate the overlap percentage between a transcription segment and a diarization segment.

        Args:
            segment (dict): The transcription segment.
            diarization (dict): The diarization segment.

        Returns:
            float: The overlap percentage between the transcription segment and the diarization segment.
        """
        start = max(transcription["start_time"], diarization["start_time"])
        end = min(transcription["end_time"], diarization["end_time"])
        overlap = max(0, end - start)
        transcription_duration = transcription["end_time"] - transcription["start_time"]
        return (overlap / transcription_duration) * 100
    
    merged_segments = []

    for d in diarization_results:
        combined_text = ""
        speaker = d["speaker"]
        used_results = []

        for t in transcription_results:
            overlap = overlap_percentage(t, d)
            if overlap > 50:  # Using 50% overlap as the threshold for inclusion
                combined_text += f"{t['text']}"
                used_results.append(t)
            elif t["end_time"] < d["end_time"]:
                combined_text += f"{t['text']}"
                used_results.append(t)
            elif t == transcription_results[-1] and d == diarization_results[-1]:
                combined_text += f"{t['text']}"
                used_results.append(t)
        
        transcription_results = [t for t in transcription_results if t not in used_results]
        
        if len(combined_text) > 0:
            d["transcription"] = combined_text
            logging.info(f"Transcription for {d['speaker']}: {str(d['start_time'])} - {str(d['end_time'])} - {d['transcription']}")
            merged_segments.append(d)
        else:
            logging.info(f"No text found for {d['speaker']}: {str(d['start_time'])} - {str(d['end_time'])}")

    return merged_segments

def create_transcript(merged_segments: List[dict]) -> str:
    """
    Create a transcript from the merged segments.

    This function takes the merged segments, which contain both diarization and transcription information, and generates a formatted transcript. The transcript is created by iterating over each segment and concatenating the speaker label and the corresponding transcription text. The resulting transcript is a string where each segment is represented as:

    SPEAKER_LABEL: TRANSCRIPTION_TEXT

    Args:
        merged_segments (List[dict]): A list of dictionaries representing the merged segments. Each dictionary should have the following keys:
            - "speaker" (str): The speaker label for the segment.
            - "transcription" (str): The transcription text for the segment.

    Returns:
        str: The generated transcript as a string.

    Example:
        >>> merged_segments = [
                {
                    "speaker": "SPEAKER_00",
                    "transcription": "Hello, how are you?"
                },
                {
                    "speaker": "SPEAKER_01", 
                    "transcription": "I'm doing well, thanks for asking!"
                },
                {
                    "speaker": "SPEAKER_00",
                    "transcription": "That's great to hear."
                }
            ]
        >>> transcript = create_transcript(merged_segments)
        >>> print(transcript)
        SPEAKER_00: Hello, how are you?

        SPEAKER_01: I'm doing well, thanks for asking!

        SPEAKER_00: That's great to hear.

    """
    transcript = ""
    for segment in merged_segments:
        transcript += f"{segment['speaker']}: {segment['transcription']}\n\n"
    return transcript

def main(youtube_url: str):
    """
    Main function that orchestrates the entire process of downloading a YouTube video, splitting it into audio chunks, transcribing and diarizing the chunks, merging the results, and creating a final transcript.

    Args:
        youtube_url (str): The URL of the YouTube video to be processed.

    Returns:
        str: The final transcript of the processed YouTube video.

    The main function performs the following steps:
    1. Downloads the audio from the specified YouTube video using the `yt_dlp_download` function.
    2. Creates a temporary directory to store the audio chunks.
    3. Splits the downloaded audio into smaller chunks using the `create_audio_chunks` function.
    4. Transcribes the audio chunks using the `main_transcribe_audio` function.
    5. Diarizes the audio chunks using the `diarize_and_condense_audio_chunks` function.
    6. Deletes the temporary directory containing the audio chunks.
    7. Merges the diarization and transcription results using the `merge_diarization_and_transcription` function.
    8. Creates the final transcript using the `create_transcript` function.
    9. Returns the final transcript.

    This function serves as the entry point for the entire process, coordinating the various steps involved in processing a YouTube video and generating a transcript with speaker information.
    """
    # Download the audio file
    audio_file_path =  yt_dlp_download(youtube_url)

    # Create a temporary directory to store the audio chunks
    chunk_output_dir = "temp"
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Create audio chunks
    audio_chunk_paths = create_audio_chunks(audio_file_path, chunk_output_dir)
    # Transcribe the audio chunks
    transcription_results = main_transcribe_audio(audio_chunk_paths)
    # Diarize the audio chunks
    diarization_results = diarize_and_condense_audio_chunks(audio_chunk_paths)
    
    # Delete the temporary audio chunks directory
    shutil.rmtree(chunk_output_dir)
    
    # Merge the diarization results and transcription results
    merged_segments = merge_diarization_and_transcription(diarization_results, transcription_results)
    
    # Create the transcript
    transcript = create_transcript(merged_segments)

    return transcript

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=PX2_gTCbjao&ab_channel=NBCBayArea"
    download_path = yt_dlp_download(youtube_url)  
    create_audio_chunks(download_path, "temp")
