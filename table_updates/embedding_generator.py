from genai_toolbox.chunk_and_embed.embedding_functions import (
    create_openai_embedding, 
    embed_dict_list, 
    add_similarity_to_next_dict_item,
    embed_dict_list_async,
)
from genai_toolbox.chunk_and_embed.chunking_functions import (
    convert_utterance_speaker_to_speakers, consolidate_similar_utterances,
    add_metadata_to_chunks, format_speakers_in_utterances,
    milliseconds_to_minutes_in_utterances, rename_start_end_to_ms
)
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from genai_toolbox.helper_functions.datetime_helpers import convert_date_format

from transcript_summary import summarize_transcript, convert_summary_to_utterance

import logging
from typing import List, Dict
import re
import time

from dotenv import load_dotenv
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm
from httpx import HTTPStatusError

load_dotenv()

class HTTPXFilter(logging.Filter):
    """
        The HTTPXFilter class is a custom logging filter that is designed to control the logging output 
        of HTTP requests made using the httpx library. It inherits from the logging.Filter class and 
        overrides the filter method to selectively allow or block log records based on specific criteria.

        Attributes:
            first_request (bool): A flag that indicates whether the first HTTP request has been logged. 
                                It is initialized to True.

        Methods:
            filter(record): This method checks if the log record contains the string 'HTTP Request:'. 
                            If it does and it's the first request, the method allows the record to be logged 
                            and sets the first_request flag to False. For subsequent requests, it blocks 
                            the log record from being logged. If the log record does not contain the 
                            specified string, it allows the record to be logged.

        This class is useful for reducing log clutter by only logging the first HTTP request, which can 
        help in debugging and monitoring HTTP interactions without overwhelming the log output.
    """
    def __init__(self):
        super().__init__()
        self.first_request = True

    def filter(self, record):
        if 'HTTP Request:' in record.getMessage():
            if self.first_request:
                self.first_request = False
                return True
            return False
        return True

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").addFilter(HTTPXFilter())

async def process_entry(
    entry: Dict,
    consolidation_threshold: float = 0.35,
    pbar: tqdm = None,
    per_worker_rate_limit: int = 1000
) -> List[Dict]:
    """
        Processes a single entry from the feed, performing various operations such as 
        converting speakers, embedding utterances, processing similarities, consolidating 
        utterances, formatting times, and adding metadata.

        Function Dependencies:
            - convert_utterance_speaker_to_speakers
            - embed_dict_list_async
            - add_similarity_to_next_dict_item
            - consolidate_similar_utterances
            - format_speakers_in_utterances
            - milliseconds_to_minutes_in_utterances
            - rename_start_end_to_ms
            - add_metadata_to_chunks

        Args:
            entry (Dict): A dictionary containing the entry data, which includes:
                - feed_title (str): The title of the feed.
                - title (str): The title of the episode.
                - published (str): The published date of the episode.
                - replaced_dict (Dict): A dictionary containing transcribed utterances.
                - video_id (str, optional): The ID of the video, if available.
            consolidation_threshold (float, optional): The threshold for consolidating 
                similar utterances. Default is 0.35.
            pbar (tqdm, optional): A progress bar instance for tracking progress.
            per_worker_rate_limit (int, optional): The rate limit for embedding requests 
                per worker. Default is 1000.

        Returns:
            List[Dict]: A list of dictionaries representing the processed and consolidated 
            utterances, each containing the relevant metadata and embeddings.

        Raises:
            Exception: Raises an exception if any processing step fails, which can be 
            caught and handled by the calling function.

        This function is designed to be used in an asynchronous context and should be 
        awaited when called. It utilizes various helper functions to perform specific 
        tasks, ensuring modularity and reusability of code.
    """
    
    video_id = entry['video_id']
    feed_title = entry['feed_title']
    episode_title = entry['title']
    feed_regex = re.sub(r'[^a-zA-Z0-9\s]', '', feed_title)
    episode_regex = re.sub(r'[^a-zA-Z0-9\s]', '', episode_title)
    episode_date = convert_date_format(entry['published'])
    
    utterances = entry['replaced_dict']['transcribed_utterances']

    async def update_stage(stage_name):
        if pbar:
            pbar.set_postfix({"stage": stage_name}, refresh=True)

    await update_stage("Generating summary")
    transcript = entry['replaced_dict']['transcript']
    summary = summarize_transcript(transcript)
    summary_utterance = convert_summary_to_utterance(
        summary,
        video_id,
        episode_title,
        feed_title,
        feed_regex,
        episode_regex,
        episode_date
    )

    await update_stage("Embedding summary")
    summary_embedding = await embed_dict_list_async(
        embedding_function=create_openai_embedding,
        chunk_dicts=[summary_utterance], 
        key_to_embed="text",
        model_choice="text-embedding-3-large",
        rate_limit=per_worker_rate_limit
    )

    await update_stage("Converting speakers")
    speakermod_utterances = convert_utterance_speaker_to_speakers(utterances)
    
    await update_stage("Embedding utterances")
    embedded_utterances = await embed_dict_list_async(
        embedding_function=create_openai_embedding,
        chunk_dicts=speakermod_utterances, 
        key_to_embed="text",
        model_choice="text-embedding-3-large",
        rate_limit=per_worker_rate_limit
    )
    
    await update_stage("Processing similarities")
    similar_utterances = add_similarity_to_next_dict_item(embedded_utterances)
    filtered_utterances = [
        {k: v for k, v in utterance.items() if k != 'embedding'}
        for utterance in similar_utterances
    ]
    
    await update_stage("Consolidating utterances")
    consolidated_similar_utterances = consolidate_similar_utterances(
        filtered_utterances, 
        similarity_threshold=consolidation_threshold
    )

    await update_stage("Formatting times utterances")
    formatted_utterances = format_speakers_in_utterances(consolidated_similar_utterances)
    minutes_utterances = milliseconds_to_minutes_in_utterances(formatted_utterances)
    renamed_utterances = rename_start_end_to_ms(minutes_utterances)

    await update_stage("Adding metadata")
    additional_metadata = {
        "title": f"{feed_title} - {episode_date}: {episode_title}",
        "publisher": feed_title,
        "date_published": episode_date,
    }
    titled_utterances = add_metadata_to_chunks(
        chunks=renamed_utterances,
        additional_metadata=additional_metadata
    )
    
    await update_stage("Embedding consolidated utterances")
    consolidated_embeddings = await embed_dict_list_async(
        embedding_function=create_openai_embedding,
        chunk_dicts=titled_utterances, 
        key_to_embed="text",
        model_choice="text-embedding-3-large",
        rate_limit=per_worker_rate_limit
    )

    await update_stage("Creating YouTube links")
    if video_id:
        for utterance in consolidated_embeddings:
            start_seconds = utterance['start_ms'] // 1000
            utterance['youtube_link'] = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
    
    await update_stage("Generating IDs")
    for utterance in consolidated_embeddings:
        start = utterance['start_ms']
        utterance['id'] = f"{start} {feed_regex} {episode_regex}".replace(' ', '-')

    await update_stage("Adding summary embedding")
    consolidated_embeddings.append(summary_embedding[0])
    
    await update_stage("Completed")
    return consolidated_embeddings

async def process_entry_async(
    entry: Dict, 
    semaphore: Semaphore,
    pbar: tqdm,
    per_worker_rate_limit: int = 1000
) -> List[Dict]:
    """
        Processes a single entry asynchronously, updating the progress bar and logging the processing time.

        Function Dependencies:
            - process_entry

        Args:
            entry (Dict): The entry to be processed, expected to contain at least a 'title' key.
            semaphore (Semaphore): A semaphore to limit the number of concurrent tasks.
            pbar (tqdm): A progress bar instance to update the processing status.
            per_worker_rate_limit (int): The rate limit for processing per worker.

        Returns:
            List[Dict]: The result of processing the entry, or None if an error occurred.
    """
    async with semaphore:
        try:
            start_time = time.time()
            entry_title = entry['title'][:30]
            logging.info(f"Started processing {entry_title}...")
            pbar.set_description(f"Processing {entry_title}...")
            result = await process_entry(
                entry, 
                pbar=pbar,
                per_worker_rate_limit=per_worker_rate_limit
            )
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Finished processing {entry_title} in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            logging.error(f"Error processing entry {entry['title']}: {str(e)}")
            return None

def load_existing_dicts(
    embedding_config: Dict
) -> List[Dict]:
    """
        Loads existing dictionaries from a file and returns them.

        Args:
            embedding_config (Dict): A dictionary containing configuration details for loading existing dictionaries.

        Returns:
            List[Dict]: A list of dictionaries loaded from the file.
    """
    try:
        return retrieve_file(
            file_name=embedding_config['existing_embeddings_file'],
            dir_name=embedding_config['existing_embeddings_dir']
        )
    except FileNotFoundError:
        logging.info("No existing aggregated chunked embeddings found. Creating new file.")
        return []

async def generate_embeddings_async(
    embedding_config: Dict,
    include_existing: bool = False,
    max_concurrent_tasks: int = 5
) -> List[Dict]:
    """
        Asynchronously generates embeddings for a list of entries based on the provided configuration.

        Function Dependencies:
            - process_entry_async
            - retrieve_file
            - write_to_file

        Args:
            embedding_config (Dict): A dictionary containing configuration details for generating embeddings, 
                including input file paths and output file names.
            include_existing (bool, optional): A flag indicating whether to include existing embeddings in the 
                processing. Default is False.
            max_concurrent_tasks (int, optional): The maximum number of concurrent tasks to run for processing 
                entries. Default is 5.

        Returns:
            List[Dict]: A list of dictionaries containing the generated embeddings for each entry processed. 
                If an error occurs during processing, an empty list is returned.

        This function retrieves the input data from the specified file, processes each entry asynchronously 
        while respecting the rate limits, and writes the results to an output file. It utilizes a semaphore 
        to limit the number of concurrent tasks and a progress bar to track the processing status.
    """
    feed_dict = retrieve_file(
        file_name=embedding_config['input_file'],
        dir_name=embedding_config['input_dir']
    )

    total_rate_limit = 5000
    per_worker_rate_limit = total_rate_limit / max_concurrent_tasks

    semaphore = Semaphore(max_concurrent_tasks)

    chunked_dicts = []
    with tqdm(total=len(feed_dict), desc="Processing entries") as pbar:
        tasks = [process_entry_async(
            entry, 
            semaphore, 
            pbar, 
            per_worker_rate_limit=per_worker_rate_limit
        ) for entry in feed_dict]
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result:
                chunked_dicts.extend(result)
            pbar.update(1)

    # Add these lines to write the results to a file
    write_to_file(
        content=chunked_dicts,
        file_name=embedding_config['output_file_name'],
        dir_name=embedding_config['output_dir_name']
    )

    return chunked_dicts

def generate_embeddings(
    embedding_config: Dict,
    include_existing: bool = False,
    max_concurrent_tasks: int = 15
) -> List[Dict]:
    """
        Generates embeddings for a set of entries based on the provided configuration.

        This function serves as a wrapper for the asynchronous function 
        `generate_embeddings_async`, allowing it to be called in a synchronous context. 
        It retrieves the input data from the specified file, processes each entry 
        asynchronously while respecting the rate limits, and writes the results 
        to an output file.

        Function Dependencies:
            - generate_embeddings_async

        Args:
            embedding_config (Dict): A dictionary containing configuration settings 
                for generating embeddings, including input file paths and other 
                necessary parameters.
            include_existing (bool, optional): A flag indicating whether to include 
                existing embeddings in the processing. Default is False.
            max_concurrent_tasks (int, optional): The maximum number of concurrent 
                tasks to run for processing entries. Default is 15.

        Returns:
            List[Dict]: A list of dictionaries containing the generated embeddings 
                for each entry processed. If an error occurs during processing, 
                an empty list is returned.

        This function is designed to be used in a synchronous context and should 
        be called when you want to generate embeddings for a set of entries. 
        It utilizes asyncio to run the asynchronous processing in a blocking manner.
    """
    return asyncio.run(
        generate_embeddings_async(
            embedding_config, 
            include_existing, 
            max_concurrent_tasks
        )
    )
