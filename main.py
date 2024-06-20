from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_audio_summary
from genai_toolbox.helper_functions.string_helpers import write_string_to_file, retrieve_string_from_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_transcript, replace_assemblyai_speakers
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.text_prompting.model_calls import openai_text_response, anthropic_text_response

import json
import logging
import numpy as np
from typing import Callable

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
    print(f"QUERY RESPONSE:\n\n{query_response}\n\n")

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

if __name__ == "__main__":
    # Example usage
    feed_url = "https://feeds.megaphone.fm/HS2300184645"
    start_date_input = "June 4, 2024"
    end_date_input = "June 6, 2024"

    if False:
        feed_entries = return_entries_by_date(feed_url, start_date_input, end_date_input)
        write_string_to_file("mfm_feed.txt", json.dumps(feed_entries, indent=4))
    else:
        feed_entries = retrieve_string_from_file("mfm_feed.txt")
        feed_entries = json.loads(feed_entries)

    episode = feed_entries[0]
    episode_title = episode['title']

    if False:
        audio_file_path = download_podcast_audio(episode['url'], episode['title'])
        assemblyai_transcript = generate_assemblyai_transcript(audio_file_path, "assemblyai_transcript.txt")
        write_string_to_file("assemblyai_transcript.txt", assemblyai_transcript)
    else:
        assemblyai_transcript = retrieve_string_from_file("assemblyai_transcript.txt")


    if False:
        summary_text = generate_audio_summary(episode['summary'], episode['feed_summary'])
        replaced_transcript = replace_assemblyai_speakers(assemblyai_transcript, summary_text)
        write_string_to_file("replaced_transcript.txt", replaced_transcript)
    else:
        replaced_transcript = retrieve_string_from_file("replaced_transcript.txt")

    chunks = split_text_string(replaced_transcript, "\n\n")
    lengthened_chunks = consolidate_short_chunks(chunks, 100)

    if False:
        model_choice = "text-embedding-3-large"
        lengthened_chunk_embeddings = embed_string_list(lengthened_chunks, client, model_choice)
        write_string_to_file("embeddings.txt", json.dumps(lengthened_chunk_embeddings, indent=4))
    else:
        lengthened_chunk_embeddings = retrieve_string_from_file("embeddings.txt")
        lengthened_chunk_embeddings = json.loads(lengthened_chunk_embeddings)

    similar_chunks = consolidate_similar_chunks(lengthened_chunks, lengthened_chunk_embeddings, .5)

    if False:
        model_choice = "text-embedding-3-large"
        similar_chunk_embeddings = embed_string_list(similar_chunks, client, model_choice)
        write_string_to_file("similar_chunk_embeddings.txt", json.dumps(similar_chunk_embeddings, indent=4))
    else:
        similar_chunk_embeddings = retrieve_string_from_file("similar_chunk_embeddings.txt")
        similar_chunk_embeddings = json.loads(similar_chunk_embeddings)

    print(f"len(similar_chunks): {len(similar_chunks)}")
    print(f"len(similar_chunk_embeddings): {len(similar_chunk_embeddings)}")
    print(f"type(similar_chunks): {type(similar_chunks)}")
    print(f"type(similar_chunk_embeddings): {type(similar_chunk_embeddings[0])}")

    chunks_with_metadata = create_chunks_with_metadata(
        source=episode_title,
        chunks=similar_chunks,
        chunk_embeddings=similar_chunk_embeddings
    )

    question = "What is their advice around raising a family?"

    response = llm_response_with_query(
        question=question,
        chunks_with_metadata=chunks_with_metadata,
        llm_function=anthropic_text_response,
        llm_model="sonnet",
        max_query_chunks=8,
        threshold=0.25
    )
    print(f"\n\n{response}\n\n")