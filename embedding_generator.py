


from genai_toolbox.chunk_and_embed.embedding_functions import (
    create_openai_embedding, 
    embed_dict_list, 
    add_similarity_to_next_dict_item
)
from genai_toolbox.chunk_and_embed.chunking_functions import (
    convert_utterance_speaker_to_speakers, consolidate_similar_utterances,
    add_metadata_to_chunks, format_speakers_in_utterances,
    milliseconds_to_minutes_in_utterances, rename_start_end_to_ms
)
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from genai_toolbox.helper_functions.datetime_helpers import convert_date_format

import logging
from typing import List, Dict
import re
import time

import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO)

async def process_entry(
    entry: Dict,
    consolidation_threshold: float = 0.35,
    pbar: tqdm = None
) -> List[Dict]:
    feed_title = entry['feed_title']
    episode_title = entry['title']
    episode_date = convert_date_format(entry['published'])
    utterances = entry['replaced_dict']['transcribed_utterances']
    
    async def update_stage(stage_name):
        if pbar:
            pbar.set_postfix({"stage": stage_name}, refresh=True)

    await update_stage("Converting speakers")
    speakermod_utterances = convert_utterance_speaker_to_speakers(utterances)
    
    await update_stage("Embedding utterances")
    embedded_utterances = await asyncio.to_thread(
        embed_dict_list,
        embedding_function=create_openai_embedding,
        chunk_dicts=speakermod_utterances, 
        key_to_embed="text",
        model_choice="text-embedding-3-large"
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
    
    await update_stage("Embedding consolidated utterances")
    consolidated_embeddings = await asyncio.to_thread(
        embed_dict_list,
        embedding_function=create_openai_embedding,
        chunk_dicts=consolidated_similar_utterances, 
        key_to_embed="text",
        model_choice="text-embedding-3-large"
    )
    
    await update_stage("Adding metadata")
    additional_metadata = {
        "title": f"{feed_title} - {episode_date}: {episode_title}"
    }
    titled_utterances = add_metadata_to_chunks(
        chunks=consolidated_embeddings,
        additional_metadata=additional_metadata
    )
    
    await update_stage("Formatting utterances")
    formatted_utterances = format_speakers_in_utterances(titled_utterances)
    minutes_utterances = milliseconds_to_minutes_in_utterances(formatted_utterances)
    renamed_utterances = rename_start_end_to_ms(minutes_utterances)
    
    await update_stage("Generating IDs")
    for utterance in renamed_utterances:
        start = utterance['start_ms']
        feed_regex = re.sub(r'[^a-zA-Z0-9\s]', '', feed_title)
        episode_regex = re.sub(r'[^a-zA-Z0-9\s]', '', episode_title)
        utterance['id'] = f"{start} {feed_regex} {episode_regex}".replace(' ', '-')
    
    await update_stage("Completed")
    return renamed_utterances

async def process_entry_async(entry: Dict, semaphore: Semaphore, pbar: tqdm) -> List[Dict]:
    async with semaphore:
        try:
            start_time = time.time()
            entry_title = entry['title'][:30]
            logging.info(f"Started processing {entry_title}...")
            pbar.set_description(f"Processing {entry_title}...")
            result = await process_entry(entry, pbar=pbar)
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Finished processing {entry_title} in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            logging.error(f"Error processing entry {entry['title']}: {str(e)}")
            return None


def load_existing_dicts(embedding_config: Dict) -> List[Dict]:
    try:
        return retrieve_file(
            file=embedding_config['existing_embeddings_file'], 
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
    feed_dict = retrieve_file(
        file=embedding_config['input_podcast_file'],
        dir_name=embedding_config['input_podcast_dir']
    )

    semaphore = Semaphore(max_concurrent_tasks)

    chunked_dicts = []
    with tqdm(total=len(feed_dict), desc="Processing entries") as pbar:
        tasks = [process_entry_async(entry, semaphore, pbar) for entry in feed_dict]
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result:
                chunked_dicts.extend(result)
            pbar.update(1)

    # Add these lines to write the results to a file
    write_to_file(
        content=chunked_dicts,
        file=embedding_config['output_file_name'],
        output_dir_name=embedding_config['output_dir_name']
    )

    return chunked_dicts

def generate_embeddings(
    embedding_config: Dict,
    include_existing: bool = False,
    max_concurrent_tasks: int = 5
) -> List[Dict]:
    return asyncio.run(generate_embeddings_async(embedding_config, include_existing, max_concurrent_tasks))

if __name__ == "__main__":
    pass