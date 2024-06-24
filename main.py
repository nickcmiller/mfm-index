from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, cosine_similarity, embed_dict_list
from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_episode_summary
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_utterances, replace_speakers_in_assemblyai_utterances
from genai_toolbox.text_prompting.model_calls import openai_text_response, anthropic_text_response

import json
import os
import logging

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
        
def consolidate_similar_split_chunks(
    chunks: list[dict],
    threshold: float = 0.45,
    similarity_metric: Callable = cosine_similarity
) -> list[dict]:
    """
        Consolidates similar chunks based on their embeddings.

        Args:
            chunks (list[dict]): List of dictionaries, each containing 'text' and 'embedding' keys.
            threshold (float): Similarity threshold for consolidation.
            similarity_metric (Callable): Function to compute similarity between embeddings.

        Returns:
            list[dict]: Consolidated list of dictionaries, each with a 'text' key.
    """
    current_chunk = []
    similar_chunks = []
    used_indices = set()
    
    for i in range(len(chunks)):
        if i in used_indices:
            continue
        current_chunk = [chunks[i]['text']]
        similar_found = False
        for j in range(i + 1, len(chunks)):
            if j in used_indices:
                continue
            similarity = similarity_metric(chunks[i]['embedding'], chunks[j]['embedding'])
            if similarity > threshold:
                current_chunk.append(chunks[j]['text'])
                used_indices.add(j)
                similar_found = True
            else:
                if len(current_chunk) > 1:
                    similar_chunks.append({"text": '\n\n'.join(current_chunk)})
                break      
        if similar_found is False:
            similar_chunks.append({"text": chunks[i]['text']})
        used_indices.add(i)
    
    if len(current_chunk) > 0:
        similar_chunks.append({"text": '\n\n'.join(current_chunk)})
    
    return similar_chunks

#


def create_chunks_with_metadata(
    source: str,
    chunks: list[dict],
    additional_metadata: dict = {}
) -> list[dict]:

    if not source:
        raise ValueError("Source must be provided")

    if not all('embedding' in chunk and 'text' in chunk for chunk in chunks):
        raise ValueError("Each chunk must contain both 'embedding' and 'text' keys")

    return [
        {
            "source": source,
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            **additional_metadata,
            **{k: v for k, v in chunk.items() if k not in ["text", "embedding"]}
        } for chunk in chunks
    ]

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

    """
    Example utterance:
        {
            "confidence": 0.9069812500000001,
            "end": 8374,
            "speaker": "John Gruber",
            "start": 2000,
            "text": "Ben, it's conference time of the year, and for once the world is traveling to you."
        },
    """
    def consolidate_short_assemblyai_utterances(
        transcribed_utterances: list[dict],
        min_length: int = 75
    ) -> list[dict]:
        lengthened_utterances = []
        combine_text = []
        current_start = None
        current_speakers = []

        for utterance in transcribed_utterances:
            
            current_text = utterance['text'].strip()
            if not current_text:
                continue

            if current_start is None:
                current_start = utterance['start']

            current_speakers.append(utterance['speaker'])
            first_name_initial_in_speakers = utterance['speaker'].split(' ')[0][0]
            last_name_initial_in_speakers = utterance['speaker'].split(' ')[1][0]
            formatted_text = f"{{{len(current_speakers)}}}: {current_text}"

            if len(current_text) < min_length:
                combine_text.append(formatted_text)
            else:
                if combine_text:
                    combine_text.append(formatted_text)
                    lengthened_utterances.append({
                        "start": current_start,
                        "end": utterance['end'],
                        "speakers": list(current_speakers),
                        "text": '\n\n'.join(combine_text),
                    })
                    combine_text = []
                else:
                    lengthened_utterances.append({
                        "start": current_start,
                        "end": utterance['end'],
                        "speakers": list(current_speakers),
                        "text": current_text,
                    })
                current_speakers = []
                current_start = None

        if combine_text:
            lengthened_utterances.append({
                "start": current_start,
                "end": transcribed_utterances[-1]['end'],
                "speakers": list(current_speakers),
                "text": '\n\n'.join(combine_text)
            })
    
        return lengthened_utterances

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
        """
        for i in range(len(chunk_dicts) - 1):
            current_embedding = chunk_dicts[i]['embedding']
            next_embedding = chunk_dicts[i + 1]['embedding']
            similarity = cosine_similarity(current_embedding, next_embedding)
            chunk_dicts[i]['similarity_to_next_item'] = similarity

        # similarity_to_next_item for the last item is always 0
        chunk_dicts[-1]['similarity_to_next_item'] = 0

        return chunk_dicts

                    
    consolidated_utterances = consolidate_short_assemblyai_utterances(transcribed_utterances, min_length=100)
    print(f"Length of transcribed utterances: {len(transcribed_utterances)}")
    print(f"Length of consolidated utterances: {len(consolidated_utterances)}")

    if False:
        embedded_utterances = embed_dict_list(
            embedding_function=create_openai_embedding,
            chunk_dicts=consolidated_utterances, 
            key_to_embed="text",
            model_choice="text-embedding-3-large"
        )
        similar_utterances = add_similarity_to_next_item(embedded_utterances)
        write_to_file(
            content=similar_utterances,
            file="embedded_utterances.json",
            output_dir_name="tmp"
        )
    else:
        embedded_utterances = retrieve_file(
            file="embedded_utterances.json", 
            dir_name="tmp"
        )
    for utterance in embedded_utterances:
        print(utterance['similarity_to_next_item'])
 
