from text_models import groq_text_response, openai_text_response

import os
import logging
import traceback
import re
import assemblyai as aai
import openai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

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
            Using the context of the conversation in the transcript and the background provided by the summary, identify the participating speakers.

            Summary of the conversation:\n {summary}

            Transcript of the conversation:\n {transcript}
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

    max_tries = 5
    
    for attempt in range(max_tries):
        response = openai_text_response(prompt, system_instructions=system_prompt)
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
    import json

    if False:
        feed_url = "https://feeds.megaphone.fm/HS2300184645"
        entries = return_entries_from_feed(feed_url)
        first_entry = entries[6]  
        print(json.dumps(first_entry, indent=4))
        audio_file_path = download_podcast_audio(first_entry["url"], first_entry["title"])
        print(f"AUDIO FILE PATH: {audio_file_path}")
        with open('audio_file_path.txt', 'w') as f:
            f.write(audio_file_path)
    else:
        with open('audio_file_path.txt', 'r') as f:
            audio_file_path = f.read()
    
    if False:
        assemblyai_transcript = generate_assemblyai_transcript(audio_file_path, "assemblyai_transcript.txt")
    else:
        with open('assemblyai_transcript.txt', 'r') as f:
            assemblyai_transcript = f.read()

    if True:
        feed_url = "https://feeds.megaphone.fm/HS2300184645"
        entry = return_entries_from_feed(feed_url)[6]
        feed_summary = entry["feed_summary"]
        summary = entry["summary"]

        summary_prompt = f"""
        Use the descriptions to summarize the podcast.\n
        
        Podcast Description:\n {summary} \n

        Feed Description:\n {feed_summary} \n\n

        Describe the hosts and the guests.
        """

        generated_summary = groq_text_response(summary_prompt)
        print(generated_summary)

        replace_assemblyai_speakers(assemblyai_transcript, generated_summary, output_file_path="replaced_transcript.txt")
        
