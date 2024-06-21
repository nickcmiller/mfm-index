from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_audio_summary
from genai_toolbox.helper_functions.string_helpers import write_string_to_file, retrieve_string_from_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_transcript, replace_assemblyai_speakers
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.text_prompting.model_calls import openai_text_response, anthropic_text_response

import json
import os
import logging

import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Dict, Optional

client = openai_client()

def split_text_string(
    text: str, 
    separator: str
):
    chunks = text.split(separator)
    return [chunk for chunk in chunks if chunk]

def consolidate_short_chunks(
    chunks: list[str], 
    max_length: int = 75
) -> list[str]:
    """
    Combines consecutive chunks of text that are shorter than `max_length`.
    Chunks longer than `max_length` are added as separate entries in the result list.

    Args:
    chunks (list of str): The list of text chunks to process.
    max_length (int): The maximum length for a chunk to be considered short.

    Returns:
    list of str: A new list of chunks where short consecutive chunks have been combined.
    """
    lengthened_chunks = []
    combine_chunks = []

    for chunk in chunks:
        if len(chunk) == 0:
            continue

        if len(chunk) < max_length:
            combine_chunks.append(chunk)
        else:
            if combine_chunks:
                combine_chunks.append(chunk)
                lengthened_chunks.append('\n\n'.join(combine_chunks))
                combine_chunks = []
            else:
                lengthened_chunks.append(chunk)

    if combine_chunks:
        lengthened_chunks.append('\n\n'.join(combine_chunks))

    return lengthened_chunks
        
def create_embedding(
    chunk: str,
    client: Callable,
    model_choice: str = "text-embedding-3-small"
) -> list[float]:
    response = client.embeddings.create(
        input=chunk, 
        model=model_choice
    )
    return response.data[0].embedding

def embed_string_list(
    chunks: list[str],
    client: Callable,
    model_choice: str = "text-embedding-3-small"
) -> list[list[float]]:
    return [create_embedding(chunk, client, model_choice) for chunk in chunks]

def cosine_similarity(
    vec1: list[float], 
    vec2: list[float]
):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def consolidate_similar_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    threshold: float = 0.45,
    similarity_metric: Callable = cosine_similarity
) -> list[str]:
    current_chunk = []
    similar_chunks = []
    used_indices = set()
    
    for i in range(len(chunks)):
        if i in used_indices:
            continue
        current_chunk = [chunks[i]]
        similar_found = False
        for j in range(i + 1, len(chunks)):
            if j in used_indices:
                continue
            similarity = similarity_metric(embeddings[i], embeddings[j])
            if similarity > threshold:
                current_chunk.append(chunks[j])
                used_indices.add(j)
                similar_found = True
            else:
                if len(current_chunk) > 1:
                    similar_chunks.append('\n\n'.join(current_chunk))
                break      
        if similar_found is False:
            similar_chunks.append(chunks[i])
        used_indices.add(i)
    
    if len(current_chunk) > 0:
        similar_chunks.append('\n\n'.join(current_chunk))
    
    return similar_chunks

def create_chunks_with_metadata(
    source: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    additional_metadata: dict = {}
) -> list[dict]:
    return [
        {
            "source": source,
            "text": chunk,
            "embedding": chunk_embeddings[i],
            **additional_metadata
        } for i, chunk in enumerate(chunks)
    ]

def query_chunks_with_metadata(
    query: str,
    chunks_with_metadata: list[dict], 
    client: Callable,
    model_choice: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_returned_chunks: int = 10,
) -> list[dict]:
    query_embedding = create_embedding(query, client, model_choice)

    similar_chunks = []
    for chunk in chunks_with_metadata:
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
    chunks_with_metadata: list[dict],
    query_client: Callable = openai_client(),
    query_model: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_query_chunks: int = 3,
    llm_function: Callable = openai_text_response,
    llm_model: str = "4o",
):
    query_response = query_chunks_with_metadata(
        query=question, 
        chunks_with_metadata=chunks_with_metadata,
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

async def download_multiple_episodes_by_date(
    feed_url: str,
    start_date: str,
    end_date: str,
    download_dir_name: Optional[str] = None,
    transcript_dir_name: Optional[str] = None,
    new_transcript_dir_name: Optional[str] = None
) -> List[Dict[str, str]]:
    feed_entries = return_entries_by_date(feed_url, start_date, end_date)
    updated_entries = []

    logging.info(f"Downloading {len(feed_entries)} episodes")

    async def process_entry(entry: Dict[str, str]) -> Dict[str, str]:
        try:
            audio_file_path = await asyncio.to_thread(
                download_podcast_audio, 
                entry['url'], 
                entry['title'], 
                download_dir_name=download_dir_name
            )
            entry['audio_file_path'] = audio_file_path
            entry['audio_summary'] = await asyncio.to_thread(
                generate_audio_summary, 
                entry['summary'], 
                entry['feed_summary']
            )
            entry['raw_transcript'] = await asyncio.to_thread(
                generate_assemblyai_transcript, 
                entry['audio_file_path'], 
                output_dir_name=transcript_dir_name
            )
            entry['transcript'] = await asyncio.to_thread(
                replace_assemblyai_speakers, 
                entry['raw_transcript'], 
                entry['audio_summary'],
                output_file_name=entry['title'],
                output_dir_name=new_transcript_dir_name
            )
            return entry
        except Exception as e:
            logging.error(f"Error processing entry {entry['title']}: {str(e)}")
            return None

    with ThreadPoolExecutor() as executor:
        tasks = [asyncio.create_task(process_entry(entry)) for entry in feed_entries]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"\n\nProcessing episodes"):
            result = await task
            if result:
                updated_entries.append(result)

    logging.info(f"Successfully processed {len(updated_entries)} out of {len(feed_entries)} episodes")
    return updated_entries


async def main():
    feed_url = "https://feeds.megaphone.fm/HS2300184645"
    start_date_input = "June 1, 2024"
    end_date_input = "June 4, 2024"
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
        write_string_to_file("mfm_feed.txt", json.dumps(updated_entries, indent=4))
    else:
        updated_entries = retrieve_string_from_file("mfm_feed.txt")
        updated_entries = json.loads(updated_entries)

    
   

if __name__ == "__main__":
    # asyncio.run(main())

    # chunks = split_text_string(replaced_transcript, "\n\n")
    # lengthened_chunks = consolidate_short_chunks(chunks, 100)

    # if False:
    #     model_choice = "text-embedding-3-large"
    #     lengthened_chunk_embeddings = embed_string_list(lengthened_chunks, client, model_choice)
    #     write_string_to_file("embeddings.txt", json.dumps(lengthened_chunk_embeddings, indent=4))
    # else:
    #     lengthened_chunk_embeddings = retrieve_string_from_file("embeddings.txt")
    #     lengthened_chunk_embeddings = json.loads(lengthened_chunk_embeddings)

    # similar_chunks = consolidate_similar_chunks(lengthened_chunks, lengthened_chunk_embeddings, .5)

    # if False:
    #     model_choice = "text-embedding-3-large"
    #     similar_chunk_embeddings = embed_string_list(similar_chunks, client, model_choice)
    #     write_string_to_file("similar_chunk_embeddings.txt", json.dumps(similar_chunk_embeddings, indent=4))
    # else:
    #     similar_chunk_embeddings = retrieve_string_from_file("similar_chunk_embeddings.txt")
    #     similar_chunk_embeddings = json.loads(similar_chunk_embeddings)