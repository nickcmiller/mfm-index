from groq import Groq
from text_models import groq_text_response, openai_text_response

import os
import re
from typing import List, Optional, Dict, Any
import logging
import traceback
import shutil
from dotenv import load_dotenv
import time
import httpx
import json

import assemblyai as aai

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def call_groq(
    audio_file: str
) -> Dict[str, Any]:
    """
    This function transcribes an audio file using the Groq library.

    Args:
        audio_file (str): The path to the audio file to be transcribed.

    Returns:
        str: JSON with the transcribed audio.
    """
    # client = Groq()
    from openai import OpenAI
    client = OpenAI()
    max_retries = 6
    retry_delay = 10  # seconds

    # Transcribe the audio file
    try:
        # Get the file size in megabytes and log it
        file_size_bytes = os.path.getsize(audio_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        logging.info(f"The size of the file sent to Groq Whisper is {file_size_megabytes} MB.")

        for attempt in range(max_retries):
            try:
                with open(audio_file, "rb") as af:
                    transcription= client.audio.transcriptions.create(
                        file=(audio_file, af.read()),
                        # model="whisper-large-v3",
                        model="whisper-1",
                        response_format="verbose_json"
                    )
                time.sleep(3)
                return transcription
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
        logging.error(f"File {audio_file} not found.")
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

def transcribe_chunks(
    audio_chunk_paths: List[str], 
    temp_dir: str
) -> List[dict]:
    """
    Transcribes a list of audio chunks and returns the transcription results.

    This function takes a list of file paths to audio chunks and a temporary directory path. It iterates over each audio chunk, calls the `call_groq` function to transcribe the chunk, and appends the transcription result to the `transcribed_chunks` list. 

    The transcription result for each chunk is a dictionary containing the following keys:
    - "segments": The transcribed segments returned by the `call_groq` function.
    - "duration": The duration of the audio chunk.

    If an exception occurs during the transcription process for a chunk, an error message is logged along with the traceback.

    Args:
        audio_chunk_paths (List[str]): A list of file paths to the audio chunks to be transcribed.
        temp_dir (str): The path to the temporary directory where the audio chunks are stored.

    Returns:
        List[dict]: A list of dictionaries representing the transcription results for each audio chunk. Each dictionary contains the "segments" and "duration" keys.

    Example:
        >>> audio_chunk_paths = ["chunk1.mp3", "chunk2.mp3", "chunk3.mp3"]
        >>> temp_dir = "/tmp/audio_chunks"
        >>> transcribed_chunks = transcribe_chunks(audio_chunk_paths, temp_dir)
        >>> print(transcribed_chunks)
        [
            {
                "segments": [{"text": "Hello world", "start": 0.0, "end": 1.5}, ...],
                "duration": 25.0
            },
            {
                "segments": [{"text": "This is the second chunk", "start": 0.0, "end": 2.3}, ...],
                "duration": 25.0
            },
            {
                "segments": [{"text": "Final chunk of audio", "start": 0.0, "end": 3.1}, ...],
                "duration": 20.0
            }
        ]
    """
    transcribed_chunks = []

    for chunk_path in audio_chunk_paths:
        try:
            response = call_groq(chunk_path)
        except Exception as e:
            logging.error(f"transcribe_chunks failed to call_groq for {chunk_path}: {e}")
            logging.error(traceback.format_exc())
            response = None

        transcribed_chunk = {
            "segments": response.segments,
            "duration": response.duration
        }
        transcribed_chunks.append(transcribed_chunk)
    
    return transcribed_chunks

def clump_response(
    segments: List[dict], 
    added_duration: float=0.0
) -> List[dict]:
    """
    Clumps together transcription segments into longer segments based on punctuation and duration.

    This function takes a list of transcription segments and combines them into longer segments based on the following rules:
    1. If a segment ends with a punctuation mark (period, question mark, or exclamation point), it is considered a complete segment and added to the formatted_segments list.
    2. If a segment does not end with punctuation but the total duration of the current clumped segment exceeds 60 seconds, each individual segment is added to the formatted_segments list as is.
    3. If a segment does not end with punctuation and the total duration is less than 60 seconds, it is combined with the next segment.
    4. The start_time and end_time of each formatted segment are adjusted by adding the added_duration parameter to account for the duration of previous audio chunks.

    Args:
        segments (List[dict]): A list of dictionaries representing the transcription segments. Each dictionary should have the following keys:
            - "start": The start time of the segment in seconds.
            - "end": The end time of the segment in seconds. 
            - "text": The transcription text of the segment.
        added_duration (float): The duration in seconds to add to the start and end times of each segment. This is used when combining segments from multiple audio chunks.

    Returns:
        List[dict]: A list of dictionaries representing the clumped segments. Each dictionary has the following keys:
            - "start_time": The adjusted start time of the segment in seconds.
            - "end_time": The adjusted end time of the segment in seconds.
            - "text": The transcription text of the clumped segment.

    Example:
        >>> segments = [
                {"start": 0.0, "end": 2.5, "text": "This is the first segment"},
                {"start": 2.5, "end": 5.0, "text": " and this is the second part."},
                {"start": 5.0, "end": 7.0, "text": "A new sentence starts here"},
                {"start": 7.0, "end": 10.0, "text": " but it doesn't end with punctuation"},
                {"start": 10.0, "end": 12.5, "text": " so it continues until this part!"}
            ]
        >>> clumped_segments = clump_response(segments, added_duration=30.0)
        >>> print(clumped_segments)
        [
            {
                "start_time": 30.0, 
                "end_time": 35.0,
                "text": "This is the first segment and this is the second part."
            },
            {
                "start_time": 35.0,
                "end_time": 42.5, 
                "text": "A new sentence starts here but it doesn't end with punctuation so it continues until this part!"
            },  
        ]
    """
    formatted_segments = []
    current_segment = ""
    current_segments = []
    start_time = 0
    end_time = 0

    for segment in segments:
        # If the current segment is empty, add the text to the current segment
        if current_segment == "":
            current_segment += segment["text"]
            current_segments.append(segment)
            start_time = segment["start"]
        # If the current segment ends with a punctuation mark, add the current segment to the formatted segments
        elif segment["text"].strip().endswith(('.', '?', '!')) or segment["end"] == segments[-1]["end"]:
            end_time = segment["end"]
            if end_time - start_time > 60:
                for s in current_segments:
                    formatted_segments.append({
                        "start_time": s["start"] + added_duration,
                        "end_time": s["end"] + added_duration,
                        "text": s["text"]
                    })
            else:
                current_segment += segment["text"]
                formatted_segments.append({
                    "start_time": start_time + added_duration, 
                    "end_time": end_time + added_duration, 
                    "text": current_segment
                })
            current_segment = ""
            current_segments = []
        else:
            current_segment += segment["text"]
            current_segments.append(segment)

    return formatted_segments

def default_response(
    segments: List[dict], 
    added_duration: float=0.0
) -> List[dict]:
    """
    This function takes a list of segments and an optional added_duration, and returns a list of formatted segments.

    The default_response function simply formats each segment by adding the added_duration to the start and end times, and returns a list of dictionaries containing the formatted segments.

    Args:
        segments (List[dict]): A list of dictionaries representing the segments. Each dictionary should have the following keys:
            - "start": The start time of the segment in seconds.
            - "end": The end time of the segment in seconds.
            - "text": The text content of the segment.
        added_duration (float, optional): The duration to be added to the start and end times of each segment. Defaults to 0.0.

    Returns:
        List[dict]: A list of dictionaries representing the formatted segments. Each dictionary has the following keys:
            - "start_time": The adjusted start time of the segment in seconds.
            - "end_time": The adjusted end time of the segment in seconds.
            - "text": The text content of the segment.

    Example:
        >>> segments = [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is the first segment."
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "This is the second segment."
                }
            ]
        >>> formatted_segments = default_response(segments, added_duration=30.0)
        >>> print(formatted_segments)
        [
            {
                "start_time": 30.0,
                "end_time": 35.0,
                "text": "This is the first segment."
            },
            {
                "start_time": 35.0,
                "end_time": 40.0,
                "text": "This is the second segment."
            }
        ]
    """
    formatted_segments = []
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        formatted_segments.append({
                "start_time": start_time + added_duration, 
                "end_time": end_time + added_duration, 
                "text": text
        })

    return formatted_segments

def format_chunks(
    transcribed_chunks: List[dict], 
    response_type: str="default"
) -> List[dict]:
    """
    Format the transcribed chunks based on the specified response type.

    This function takes the transcribed chunks and formats them based on the specified response type. It supports two response types:
    - "clump": Combines the segments within each chunk into a single segment.
    - "default" (or any other value): Keeps the segments within each chunk separate and adjusts their start and end times based on the duration of the previous chunks.

    Args:
        transcribed_chunks (List[dict]): A list of dictionaries representing the transcribed chunks. Each dictionary should have the following keys:
            - "segments": A list of dictionaries representing the segments within the chunk. Each segment dictionary should have the keys "start", "end", and "text".
            - "duration": The duration of the chunk in seconds.
        response_type (str): The type of response to generate. Can be "clump" or "default". Defaults to "default".

    Returns:
        List[dict]: A list of dictionaries representing the formatted segments. Each dictionary has the following keys:
            - "start_time": The start time of the segment in seconds.
            - "end_time": The end time of the segment in seconds.
            - "text": The text content of the segment.

    Example:
        >>> transcribed_chunks = [
                {
                    "segments": [
                        {"start": 0.0, "end": 2.5, "text": "This is the first segment of chunk 1."},
                        {"start": 2.5, "end": 5.0, "text": "This is the second segment of chunk 1."}
                    ],
                    "duration": 5.0
                },
                {
                    "segments": [
                        {"start": 0.0, "end": 3.0, "text": "This is the first segment of chunk 2."},
                        {"start": 3.0, "end": 6.0, "text": "This is the second segment of chunk 2."}
                    ],
                    "duration": 6.0
                }
            ]
        >>> formatted_segments = format_chunks(transcribed_chunks, response_type="clump")
        >>> print(formatted_segments)
        [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "This is the first segment of chunk 1. This is the second segment of chunk 1."
            },
            {
                "start_time": 5.0,
                "end_time": 11.0,
                "text": "This is the first segment of chunk 2. This is the second segment of chunk 2."
            }
        ]
    """
    duration = 0
    segments = []

    for chunk_dict in transcribed_chunks:
        if response_type == "clump":
            formatted_response = clump_response(chunk_dict["segments"], duration)
        else:
            formatted_response = default_response(chunk_dict["segments"], duration)
        segments.extend(formatted_response)
        duration += chunk_dict["duration"]

    return segments

def main_transcribe_audio(
    audio_chunk_paths: List[str], 
    response_type: str="clump"
) -> List[dict]:
    """
        Main function to transcribe audio chunks and format the transcription results.

        This function takes a list of file paths to audio chunks and a response type (default is "clump"). It performs the following steps:
        1. Transcribes the audio chunks using the `transcribe_chunks` function.
        2. Formats the transcribed chunks based on the specified response type using the `format_chunks` function.
        3. Returns the formatted transcription segments.

        Args:
            audio_chunk_paths (List[str]): A list of file paths to the audio chunks to be transcribed.
            response_type (str, optional): The type of response formatting to apply. Default is "clump".

        Returns:
            List[dict]: A list of dictionaries representing the formatted transcription segments. The format of the segments depends on the `response_type` argument.

        Example:
        >>> audio_chunk_paths = ["01_chunk.mp3", "02_chunk.mp3"]
        >>> response_type = "clump"
        >>> transcription_segments = main_transcribe_audio(audio_chunk_paths, response_type)
        >>> print(transcription_segments)
        [
            {
                "start_time": 0.0,
                "end_time": 20.0,
                "text": "This is the transcription for chunk 1. It contains the text from all the segments in chunk 1."
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "text": "This is part 1 of chunk 2."
            },
            {
                "start_time": 30.0,
                "end_time": 40.0,
                "text": "This is part 2 of chunk 2."
            }
        ]
    """
    transcribed_chunks = transcribe_chunks(audio_chunk_paths, "temp")
    segments = format_chunks(transcribed_chunks, response_type=response_type)
    
    return segments


# AssemblyAI Functions

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_audio_assemblyai(
    audio_file_path: str
) -> aai.transcriber.Transcript:
    """
        Transcribes an audio file using the AssemblyAI library.

        Args:
            audio_file_path (str): The path to the audio file to transcribe.

        Returns:
            aai.transcriber.Transcript: The transcription response from AssemblyAI.

        Raises:
            FileNotFoundError: If the audio file cannot be found.
            IOError: If there is an issue with reading the audio file.
            RuntimeError: If transcription fails due to API errors.

        Example of response format:
            {
                "utterances": [
                    {
                        "confidence": 0.7246,
                        "end": 3738,
                        "speaker": "A",
                        "start": 570,
                        "text": "Um hey, Erica.",
                        "words": [...]
                    },
                    {
                        "confidence": 0.6015,
                        "end": 4430,
                        "speaker": "B",
                        "start": 3834,
                        "text": "One in.",
                        "words": [...]
                    }
                ]
            }
    """
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file does not exist: {audio_file_path}")
        raise FileNotFoundError(f"Audio file does not exist: {audio_file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)

    try:
        transcriber = aai.Transcriber()
        response = transcriber.transcribe(audio_file_path, config=config)
        logging.info(f"Transcription successful for file: {audio_file_path}")
        return response
    except aai.exceptions.APIError as api_error:
        logging.error(f"API error during transcription: {api_error}")
        raise RuntimeError(f"API error: {api_error}")
    except Exception as e:
        logging.error(f"Unexpected error during transcription: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Unexpected error during transcription: {e}")

def get_transcript_assemblyai(
    response: aai.transcriber.Transcript
) -> str:
    """
        Extracts and formats the transcript from the AssemblyAI transcription response.

        Args:
            response (aai.transcriber.Transcript): The transcription response object from AssemblyAI.

        Returns:
            str: A formatted transcript string where each utterance is prefixed with the speaker label.

        Raises:
            ValueError: If the response is malformed or missing necessary data components.

        Example:
            >>> response = aai.transcriber.Transcript(...)
            >>> print(get_transcript_assemblyai(response))
            Speaker A: Hello, how are you?
            Speaker B: I'm good, thanks!
    """
    try:
        if not hasattr(response, 'utterances') or not response.utterances:
            logging.error("Invalid response: Missing 'utterances' attribute.")
            raise ValueError("Invalid response: Missing 'utterances' attribute.")

        transcript_parts = [
            f"Speaker {utterance.speaker}: {utterance.text}\n\n" for utterance in response.utterances
        ]
        return ''.join(transcript_parts)
    except Exception as e:
        logging.error(f"Failed to generate transcript: {e}")
        raise ValueError(f"Failed to generate transcript due to an error: {e}")

def evaluate_and_validate_response(
    response: str, 
    expected_type: type
) -> dict:
    """
        Evaluates and validates the response from the API.

        Args:
            response (str): The raw string response from the API.
            expected_type (type): The expected data type of the response after evaluation.

        Returns:
            dict: The evaluated response as a dictionary if it matches the expected type.

        Raises:
            SyntaxError: If there is a syntax error when evaluating the response.
            ValueError: If the evaluated response is not of the expected type.
            Exception: For any other unexpected errors during evaluation.

        Example:
            >>> response = "```python\n{'speaker_1': 'John', 'speaker_2': 'Jane'}\n```"
            >>> result = evaluate_and_validate_response(response, dict)
            >>> print(result)
            {'speaker_1': 'John', 'speaker_2': 'Jane'}
    """
    try:
        # Attempt to evaluate the response directly
        eval_response = eval(response)
        if isinstance(eval_response, expected_type):
            logging.info("Response successfully evaluated and validated.")
            return eval_response
        else:
            raise ValueError("Response is not of the expected type.")
    except SyntaxError as e:
        logging.error(f"Syntax error during evaluation: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during evaluation: {e}")
        raise

def clean_and_validate_response(
    response: str, 
    expected_type: type
) -> dict:
    """
        Cleans and validates the response from the API.

        This function removes any code block markers (e.g., ```python) from the response string,
        and then passes the cleaned response to the evaluate_and_validate_response function
        to check if it matches the expected data type.

        Args:
            response (str): The raw string response from the API.
            expected_type (type): The expected data type of the response after cleaning and evaluation.

        Returns:
            dict: The cleaned and validated response as a dictionary if it matches the expected type.

        Raises:
            SyntaxError: If there is a syntax error when evaluating the cleaned response.
            ValueError: If the evaluated response is not of the expected type.
            Exception: For any other unexpected errors during evaluation.

        Example:
            >>> response = "```python\n{'speaker_1': 'John', 'speaker_2': 'Jane'}\n```"
            >>> result = clean_and_validate_response(response, dict)
            >>> print(result)
            {'speaker_1': 'John', 'speaker_2': 'Jane'}
    """
    cleaned_response = re.sub(r'```[\w]+', '', response).replace('```', '').strip()
    return evaluate_and_validate_response(cleaned_response, expected_type)

def identify_speakers(
    summary: str,
    transcript: str,
    prompt: str = None,
    system_prompt: str = None,
) -> dict:
    """
        Identifies the speakers in a podcast based on the summary and transcript.

        This function takes a summary and transcript of a podcast as input, along with optional prompt
        and system prompt strings. It then uses the OpenAI API to generate a response that identifies
        the speakers in the podcast. The response is expected to be a dictionary mapping speaker labels
        (e.g., "Speaker A") to their actual names.

        The function will attempt to generate a valid response up to `max_tries` times. If a valid
        dictionary response is not obtained after the maximum number of attempts, an error is logged.

        Args:
            summary (str): A summary of the podcast.
            transcript (str): The full transcript of the podcast.
            prompt (str, optional): The prompt string to use for generating the response.
                If not provided, a default prompt will be used.
            system_prompt (str, optional): The system prompt string to use for generating the response.
                If not provided, a default system prompt will be used.

        Returns:
            dict: A dictionary mapping speaker labels to their actual names.

        Example:
            >>> summary = "In this podcast, John and Jane discuss their favorite movies."
            >>> transcript = "Speaker A: My favorite movie is The Shawshank Redemption. Speaker B: Mine is Forrest Gump."
            >>> result = identify_speakers(summary, transcript)
            >>> print(result)
            {'Speaker A': 'John', 'Speaker B': 'Jane'}
    """
    if prompt is None:
        prompt = f"""
            Using the context of the conversation in the transcript and the background provided by the summary, identify the speakers in the podcast.

            Summary of the podcast:\n {summary}

            Transcript of the podcast:\n {transcript}
        """

    if system_prompt is None:
        system_prompt = """
            You only return properly formatted key-value store. 
            The output should Python eval to a dictionary. type(eval(response)) == dict

            Output Examples

            Example 1:
            {
                "Speaker A": "FirstName LastName", 
                "Speaker B": "FirstName LastName"
            }

            Example 2:
            {
                "Speaker A": "FirstName LastName", 
                "Speaker B": "FirstName LastName"
                    }
        """ 

    system_instructions = {"role": "system", "content": system_prompt}

    max_tries = 5
    
    for attempt in range(max_tries):
        response = openai_text_response(prompt, [system_instructions])
        logging.info(f"Attempt {attempt + 1}: Response received.")

        try:
            return evaluate_and_validate_response(response, dict)
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")

        try:
            return clean_and_validate_response(response, dict)
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed after cleaning: {e}")

        logging.info(f"Retrying... ({max_tries - attempt - 1} attempts left)")

    logging.error("Failed to obtain a valid dictionary response after maximum attempts.")
    raise ValueError("Failed to obtain a valid dictionary response after maximum attempts.")

def replace_speakers(
    transcript: str, 
    response_dict: dict
) -> str:
    """
        Replaces placeholders in the transcript with actual speaker names from the response dictionary.

        Args:
            transcript (str): The original transcript text containing placeholders.
            response_dict (dict): A dictionary mapping placeholders to speaker names.

        Returns:
            str: The transcript with placeholders replaced by speaker names.

        Raises:
            ValueError: If a key or value in response_dict is not a string.
    """
    for key, value in response_dict.items():
        if not isinstance(key, str) or not isinstance(value, str):
            logging.error(f"Key or value is not a string: key={key}, value={value}")
            raise ValueError(f"Key or value is not a string: key={key}, value={value}")
        
        transcript = transcript.replace(key, value)

    return transcript

def generate_assemblyai_transcript(
    audio_file_path: str, 
    output_file_path: str = None
) -> str:
    """
        Generates a transcript from an audio file using AssemblyAI and optionally writes it to a file.

        Args:
            audio_file_path (str): The path to the audio file to be transcribed.
            output_file_path (str, optional): The path to the output file where the transcript will be saved. If None, the transcript is not written to a file.

        Returns:
            str: The transcript generated from the audio file.

        Raises:
            Exception: If transcription or file writing fails.
    """
    try:
        transcribed_audio = transcribe_audio_assemblyai(audio_file_path)
        assemblyai_transcript = get_transcript_assemblyai(transcribed_audio)
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        raise Exception(f"Transcription failed for file {audio_file_path}: {e}")

    if output_file_path is not None:
        try:
            with open(output_file_path, 'w') as f:
                f.write(assemblyai_transcript)
            logging.info(f"Transcript successfully written to {output_file_path}")
        except IOError as e:
            logging.error(f"Failed to write transcript to file {output_file_path}: {e}")
            raise Exception(f"Failed to write transcript to file {output_file_path}: {e}")

    return assemblyai_transcript

def replace_assemblyai_speakers(
    assemblyai_transcript: str,
    audio_summary: str,
    first_host_speaker: str = None,
    output_file_path: str = None
) -> str:
    """
        Replaces speaker placeholders in the transcript with actual names using a summary and optionally writes the result to a file.

        Args:
            assemblyai_transcript (str): The transcript text with speaker placeholders.
            audio_summary (str): A summary of the audio content used to identify speakers.
            first_host_speaker (str, optional): The name of the first host to speak, if known.
            output_file_path (str, optional): The path to the output file where the modified transcript will be saved. If None, the transcript is not written to a file.

        Returns:
            str: The transcript with speaker placeholders replaced by actual names.

        Raises:
            Exception: If an error occurs during the processing.
    """
    try:
        if first_host_speaker:
            audio_summary += f"\n\nThe first host to speak is {first_host_speaker}"
        
        speaker_dict = identify_speakers(audio_summary, assemblyai_transcript)
        transcript_with_replaced_speakers = replace_speakers(assemblyai_transcript, speaker_dict)

        if output_file_path:
            with open(output_file_path, 'w') as f:
                f.write(transcript_with_replaced_speakers)
            logging.info(f"Transcript successfully written to {output_file_path}")

        return transcript_with_replaced_speakers
    except Exception as e:
        logging.error(f"Error in replace_assemblyai_speakers: {e}")
        raise Exception(f"Failed to process transcript: {e}")


if __name__ == "__main__":
    from podcast_functions import return_entries_from_feed, download_podcast_audio
    
    if True:
        feed_url = "https://feeds.megaphone.fm/HS2300184645"
        entries = return_entries_from_feed(feed_url)
        first_entry = entries[5]  
        print(json.dumps(first_entry, indent=4))
        audio_file_path = download_podcast_audio(first_entry["url"], first_entry["title"])
        print(f"AUDIO FILE PATH: {audio_file_path}")
        with open('audio_file_path.txt', 'w') as f:
            f.write(audio_file_path)
    else:
        with open('audio_file_path.txt', 'r') as f:
            audio_file_path = f.read()
    
    if True:
        assemblyai_transcript = generate_assemblyai_transcript(audio_file_path, "assemblyai_transcript.txt")
    else:
        with open('assemblyai_transcript.txt', 'r') as f:
            assemblyai_transcript = f.read()

    if True:
        feed_url = "https://feeds.megaphone.fm/HS2300184645"
        entry = return_entries_from_feed(feed_url)[0]
        feed_summary = entry["feed_summary"]
        summary = entry["summary"]

        summary_prompt = f"""
        Use the descriptions to summarize the podcast.\n
        
        Podcast Description:\n {summary} \n

        Feed Description:\n {feed_summary} \n\n

        Describe the hosts and the guests.
        """

        generated_summary = groq_text_response(summary_prompt)

        replace_assemblyai_speakers(assemblyai_transcript, generated_summary, "Sam Parr", "replaced_transcript.txt")
        
