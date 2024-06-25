from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, cosine_similarity, embed_dict_list
from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_episode_summary
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_utterances, replace_speakers_in_assemblyai_utterances
from genai_toolbox.text_prompting.model_calls import openai_text_response, anthropic_text_response

import json
import os
import logging
import re

import asyncio
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor

from tqdm.asyncio import tqdm
import numpy as np
from typing import Callable, List, Dict, Optional, AsyncIterator
import time

client = openai_client()
# Splitting Transcript into Chunks
def split_text_string(
    text: str, 
    separator: str
) -> list[dict]:
    """
        Splits the text by the given separator and returns a list of dictionaries where each dictionary has a key 'text' with non-empty chunks as values.

        Args:
        text (str): The text to split.
        separator (str): The separator to use for splitting the text.

        Returns:
        list[dict]: A list of dictionaries with the key 'text' and values as non-empty text chunks.
    """
    chunks = text.split(separator)
    return [{"text": chunk} for chunk in chunks if chunk]

def consolidate_split_chunks(
    chunk_dicts: list[dict], 
    min_length: int = 75
) -> list[dict]:
    """
        Combines consecutive chunks of text that are shorter than `max_length`.
        Chunks longer than `max_length` are added as separate entries in the result list.

        Args:
        chunk_dicts (list of dict): The list of dictionaries containing text chunks to process.
        max_length (int): The maximum length for a chunk to be considered short.

        Returns:
        list of dict: A new list of dictionaries where short consecutive chunks have been combined.
    """
    lengthened_chunk_dicts = []
    combine_chunks = []

    for chunk_dict in chunk_dicts:
        chunk = chunk_dict['text']
        if len(chunk) == 0:
            continue

        if len(chunk) < min_length:
            combine_chunks.append(chunk)
        else:
            if combine_chunks:
                combine_chunks.append(chunk)
                lengthened_chunk_dicts.append({"text": '\n\n'.join(combine_chunks)})
                combine_chunks = []
            else:
                lengthened_chunk_dicts.append({"text": chunk})

    if combine_chunks:
        lengthened_chunk_dicts.append({"text": '\n\n'.join(combine_chunks)})

    return lengthened_chunk_dicts
        

def add_metadata_to_chunks(
    chunks: list[dict],
    additional_metadata: dict = {}
) -> list[dict]:

    if not source:
        raise ValueError("Source must be provided")

    if not all('embedding' in chunk and 'text' in chunk for chunk in chunks):
        raise ValueError("Each chunk must contain both 'embedding' and 'text' keys")

    return [
        {    
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            **additional_metadata,
            **{k: v for k, v in chunk.items() if k not in ["text", "embedding"]}
        } for chunk in chunks
    ]

# Chunking utterances

def convert_speaker_to_speakers(
    utterances: list[dict]
) -> list[dict]:

    mod_utterances = []
    for utterance in utterances:
        speakers = [utterance['speaker']]
        mod_utterances.append({
            **{k: v for k, v in utterance.items() if k != 'speaker'},
            "speakers": speakers
        })
        
    return mod_utterances


def consolidate_short_assemblyai_utterances(
    utterances: list[dict],
    min_length: int = 75
) -> list[dict]:
    """
        Consolidates short utterances from AssemblyAI transcription into longer segments.

            Args:
                utterances (list[dict]): A list of dictionaries, each representing an utterance
                    with keys 'confidence', 'end', 'speaker', 'start', and 'text'.
            min_length (int, optional): The minimum length of text to be considered a
                standalone utterance. Defaults to 75 characters.

        Returns:
            list[dict]: A list of consolidated utterances.
    """
    consolidated = []
    current_group = None

    def finalize_group():
        if current_group:
            if len(set(current_group["speakers"])) > 1:
                text = "\n\n".join(f"{{}}: {t}" for t in current_group["texts"])
            else:
                text = current_group["texts"][0]
            
            consolidated.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "speakers": current_group["speakers"],
                "text": text,
            })

    for utterance in utterances:
        utterance_text = utterance['text'].strip()
        if not utterance_text:
            continue

        if current_group is None:
            current_group = {
                "start": utterance['start'],
                "end": utterance['end'],
                "speakers": [utterance['speaker']],
                "texts": [utterance_text],
            }
        else:
            current_group["end"] = utterance['end']
            current_group["speakers"].append(utterance['speaker'])
            current_group["texts"].append(utterance_text)

        if len(utterance_text) >= min_length:
            finalize_group()
            current_group = None

    finalize_group()  # Handle the last group if exists

    return consolidated


# Combine similar chunks

def add_similarity_to_next_item(
    chunk_dicts: list[dict],
    similarity_metric: Callable = cosine_similarity
) -> list[dict]:
    """
        Adds a 'similarity_to_next_item' key to each dictionary in the list,
        calculating the cosine similarity between the current item's embedding
        and the next item's embedding. The last item's similarity is always 0.

        Args:
            chunk_dicts (list[dict]): List of dictionaries containing 'embedding' key.

        Returns:
            list[dict]: The input list with added 'similarity_to_next_item' key for each dict.

        Example:
            Input:
            [
                {..., "embedding": [0.1, 0.2, 0.3]},
                {..., "embedding": [0.4, 0.5, 0.6]},
            ]
            Output:
            [
                {..., "embedding": [0.1, 0.2, 0.3], "similarity_to_next_item": 0.9},
                {..., "embedding": [0.4, 0.5, 0.6], "similarity_to_next_item": 0.9},
            ]
    """
    for i in range(len(chunk_dicts) - 1):
        current_embedding = chunk_dicts[i]['embedding']
        next_embedding = chunk_dicts[i + 1]['embedding']
        similarity = cosine_similarity(current_embedding, next_embedding)
        chunk_dicts[i]['similarity_to_next_item'] = similarity

    # similarity_to_next_item for the last item is always 0
    chunk_dicts[-1]['similarity_to_next_item'] = 0

    return chunk_dicts

def consolidate_similar_split_chunks(
    chunks: list[dict], 
    threshold: float = 0.45
) -> list[dict]:
    """
    Consolidates similar chunks based on their precomputed 'similarity_to_next_item'.

    Args:
        chunks (list[dict]): List of dictionaries, each containing 'text' and 'similarity_to_next_item' keys.
        threshold (float): Similarity threshold for consolidation.

    Returns:
        list[dict]: Consolidated list of dictionaries, each with a 'text' key.
    """
    consolidated_chunks = []
    current_chunk_texts = []

    for i, chunk in enumerate(chunks):
        current_chunk_texts.append(chunk['text'])

        if chunk['similarity_to_next_item'] < threshold or i == len(chunks) - 1:
            consolidated_chunks.append({"text": '\n\n'.join(current_chunk_texts)})
            current_chunk_texts = []

    return consolidated_chunks

def consolidate_similar_utterances(
    utterances: list[dict],
    similarity_threshold: float = 0.45
) -> list[dict]:
    """
        Consolidates similar utterances based on their similarity to the next item.

        Args:
            utterances (list[dict]): List of utterances, each containing 'text', 'speakers', 
                                    'similarity_to_next_item', and other fields.
            similarity_threshold (float): Threshold for considering utterances similar.

        Returns:
            list[dict]: Consolidated list of utterances.
    """
    consolidated = []
    current_group = None

    def finalize_group():
        if current_group:
            consolidated.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "speakers": current_group["speakers"],
                "text": "\n\n".join(current_group["texts"]),
            })

    def format_text(text, speakers):
        return f"{{}}: {text}" if len(speakers) == 1 else text

    for utterance in utterances:
        utterance_text = utterance['text'].strip()
        if not utterance_text:
            continue

        formatted_text = format_text(utterance_text, utterance['speakers'])

        if current_group is None:
            current_group = {
                "start": utterance['start'],
                "end": utterance['end'],
                "speakers": utterance['speakers'].copy(),
                "texts": [formatted_text],
            }
        else:
            current_group["end"] = utterance['end']
            current_group["speakers"].extend(utterance['speakers'])
            current_group["texts"].append(formatted_text)

        if utterance['similarity_to_next_item'] < similarity_threshold:
            finalize_group()
            current_group = None

    finalize_group()  # Handle the last group if exists

    return consolidated
    

# Query embeddings
def query_chunks_with_metadata(
    query: str,
    chunks_with_embeddings: list[dict], 
    client: Callable,
    model_choice: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_returned_chunks: int = 10,
) -> list[dict]:
    query_embedding = create_embedding(query, client, model_choice)

    similar_chunks = []
    for chunk in chunks_with_embeddings:
        similarity = cosine_similarity(query_embedding, chunk['embedding'])
        if similarity > threshold:
            chunk['similarity'] = similarity
            similar_chunks.append(chunk)
        
    similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    top_chunks = similar_chunks[0:max_returned_chunks]
    no_embedding_key_chunks = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in top_chunks]
    
    return no_embedding_key_chunks

def llm_response_with_query(
    question: str,
    chunks_with_embeddings: list[dict],
    query_client: Callable = openai_client(),
    query_model: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_query_chunks: int = 3,
    llm_function: Callable = openai_text_response,
    llm_model: str = "4o",
):
    query_response = query_chunks_with_metadata(
        query=question, 
        chunks_with_embeddings=chunks_with_embeddings,
        client=query_client, 
        model_choice=query_model, 
        threshold=threshold,
        max_returned_chunks=max_query_chunks
    )

    if len(query_response) == 0:
        return "Sources are not relevant enough to answer this question"

    sources = ""
    for chunk in query_response:
        sources += f"""
        Source: '{chunk['source']}',
        Text: '{chunk['text']}'
        """

    prompt = f"Question: {question}\n\nSources: {sources}"

    llm_system_prompt = f"""
    You are an expert at incorporating information from sources to provide answers. 
    Cite from the sources that are given to you in your answers.
    """
    response = llm_function(
        prompt, 
        system_instructions=llm_system_prompt, 
        model_choice=llm_model,
    )

    return response

async def download_and_transcribe_multiple_episodes_by_date(
    feed_url: str,
    start_date: str,
    end_date: str,
    download_dir_name: Optional[str] = None,
    transcript_dir_name: Optional[str] = None,
    new_transcript_dir_name: Optional[str] = None,
    max_concurrent_tasks: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, str]]:
    feed_entries = return_entries_by_date(feed_url, start_date, end_date)
    logging.info(f"Downloading {len(feed_entries)} episodes")

    async def process_entry(entry: Dict[str, str], pbar: tqdm) -> Dict[str, str]:
        for attempt in range(max_retries):
            try:
                pbar.set_description(f"Processing {entry['title'][:30]}...")
                
                entry['audio_file_path'] = await asyncio.to_thread(
                    download_podcast_audio, 
                    entry['url'], 
                    entry['title'], 
                    download_dir_name=download_dir_name
                )
                pbar.set_postfix({"stage": "audio downloaded"})
                
                entry['audio_summary'] = await asyncio.to_thread(
                    generate_episode_summary, 
                    entry['summary'], 
                    entry['feed_summary']
                )
                pbar.set_postfix({"stage": "summary generated"})
                
                entry['utterances_dict'] = await asyncio.to_thread(
                    generate_assemblyai_utterances,
                    entry['audio_file_path'], 
                    output_dir_name=transcript_dir_name
                )
                pbar.set_postfix({"stage": "raw transcript generated"})
                
                entry['transcript'] = await asyncio.to_thread(
                    replace_speakers_in_assemblyai_utterances, 
                    entry['utterances_dict'], 
                    entry['audio_summary'],
                    output_dir_name=new_transcript_dir_name
                )
                pbar.set_postfix({"stage": "transcript processed"})
                
                pbar.update(1)
                return entry
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Error processing entry {entry['title']}, attempt {attempt + 1}: {str(e)}")
                    pbar.set_postfix({"stage": f"retry {attempt + 1}/{max_retries}"})
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    logging.error(f"Failed to process entry {entry['title']} after {max_retries} attempts: {str(e)}")
                    pbar.set_postfix({"stage": "failed"})
                    pbar.update(1)
                    return None

    async with asyncio.Semaphore(max_concurrent_tasks) as semaphore:
        with tqdm(total=len(feed_entries), desc="Processing episodes") as pbar:
            tasks = [process_entry(entry, pbar) for entry in feed_entries]
            updated_entries = await asyncio.gather(*tasks)

    successful_entries = [entry for entry in updated_entries if entry is not None]
    logging.info(f"Successfully processed {len(successful_entries)} out of {len(feed_entries)} episodes")
    return successful_entries

async def main():
    feed_url = "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ"
    start_date_input = "June 1, 2024"
    end_date_input = "June 5, 2024"
    download_dir_name = "tmp_audio"
    transcript_dir_name = "tmp_transcripts"
    new_transcript_dir_name = "tmp_new_transcripts"

    if True:
        updated_entries = await download_and_transcribe_multiple_episodes_by_date(
            feed_url=feed_url,
            start_date=start_date_input,
            end_date=end_date_input,
            download_dir_name=download_dir_name,
            transcript_dir_name=transcript_dir_name,
            new_transcript_dir_name=new_transcript_dir_name
        )
        write_to_file(
            content=updated_entries,
            file="podcast_feed.txt",
            output_dir_name="tmp"
        )
    else:
        updated_entries = retrieve_file("podcast_feed.txt", dir_name="tmp")
        updated_entries = json.loads(updated_entries)

if __name__ == "__main__":
    # asyncio.run(main())

    utterances_dict = retrieve_file(
        file="Computex_2024_replaced.json", 
        dir_name="tmp_new_transcripts"
    )
    transcribed_utterances = utterances_dict['transcribed_utterances']
    speakermod_utterances = convert_speaker_to_speakers(transcribed_utterances)
    print(json.dumps(speakermod_utterances[0], indent=4))
    consolidated_utterances = consolidate_short_assemblyai_utterances(transcribed_utterances, min_length=100)
    print(json.dumps(consolidated_utterances[0], indent=4))
    # if True:
    #     embedded_utterances = embed_dict_list(
    #         embedding_function=create_openai_embedding,
    #         chunk_dicts=speakermod_utterances, 
    #         key_to_embed="text",
    #         model_choice="text-embedding-3-large"
    #     )
    #     similar_utterances = add_similarity_to_next_item(embedded_utterances)
    #     filtered_utterances = [
    #         {k: v for k, v in utterance.items() if k != 'embedding'}
    #         for utterance in similar_utterances
    #     ]
    #     write_to_file(
    #         content=filtered_utterances,
    #         file="filtered_utterances.json",
    #         output_dir_name="tmp"
    #     )
    # else:
    #     filtered_utterances = retrieve_file(
    #         file="filtered_utterances.json", 
    #         dir_name="tmp"
    #     )
    


    # consolidated_similar_utterances = consolidate_similar_utterances(filtered_utterances)

    # print(f"Length of transcribed utterances: {len(transcribed_utterances)}")
    # print(f"Length of consolidated utterances: {len(consolidated_utterances)}")
    # print(f"Length of filtered utterances: {len(filtered_utterances)}")
    # print(f"Length of consolidated similar utterances: {len(consolidated_similar_utterances)}")

    
    # if True:
    #     consolidated_embeddings = embed_dict_list(
    #         embedding_function=create_openai_embedding,
    #         chunk_dicts=consolidated_similar_utterances, 
    #         key_to_embed="text",
    #         model_choice="text-embedding-3-large"
    #     )
    #     write_to_file(
    #         content=consolidated_embeddings,
    #         file="consolidated_embeddings.json",
    #         output_dir_name="tmp"
    #     )
    # else:
    #     consolidated_embeddings = retrieve_file(
    #         file="consolidated_embeddings.json", 
    #         dir_name="tmp"
    #     )
    # print(consolidated_embeddings[0])
    

 
    # print(json.dumps(filtered_utterances, indent=4))
 
