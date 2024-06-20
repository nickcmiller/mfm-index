from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_audio_summary
from genai_toolbox.helper_functions.string_helpers import write_string_to_file, retrieve_string_from_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_transcript, replace_assemblyai_speakers
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.text_prompting.model_calls import openai_text_response

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
    return chunks

def consolidate_short_chunks(
    chunks, 
    max_length=75
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
    new_chunks = []
    combine_chunks = []

    for chunk in chunks:
        if len(chunk) < max_length:
            combine_chunks.append(chunk)
        else:
            if combine_chunks:
                combine_chunks.append(chunk)
                new_chunks.append('\n\n'.join(combine_chunks))
                combine_chunks = []
            elif len(chunk) == 0:
                continue
            else:
               new_chunks.append(chunk)

    # Ensure any remaining chunks in combine_chunks are added to new_chunks
    if len(combine_chunks) > 0:
        new_chunks.append('\n\n'.join(combine_chunks))

    return new_chunks
        

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

def query_chunks(
    query: str,
    client: Callable,
    chunks: list[str], 
    chunk_embeddings: list[list[float]], 
    model_choice: str = "text-embedding-3-small",
    threshold: float = 0.5
) -> list[str]:
    """
    Query chunks based on a text query to find similar chunks.

    Args:
    query (str): The query string.
    chunks (list of str): The list of text chunks.
    chunk_embeddings (list of list[float]): The list of embeddings corresponding to the chunks.
    threshold (float): The similarity threshold to consider a chunk as similar.

    Returns:
    list of str: A list of chunks that are similar to the query.
    """
    # Generate embedding for the query
    query_embedding = create_embedding(query, client, model_choice)

    # Find similar chunks
    similar_chunks = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > threshold:
            similar_chunks.append([chunk, similarity])

    return similar_chunks


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

    print(f"type(similar_chunk_embeddings): {type(similar_chunk_embeddings)}")
    print(f"len(chunks): {len(chunks)}")
    print(f"len(lengthened_chunks): {len(lengthened_chunks)}")
    print(f"len(similar_chunks): {len(similar_chunks)}")
    print(f"len(similar_chunk_embeddings): {len(similar_chunk_embeddings)}")

    model_choice = "text-embedding-3-large"
    query = "What are current business trends?"
    query_response = query_chunks(
        query=query, 
        client=client, 
        model_choice=model_choice, 
        chunks=similar_chunks, 
        chunk_embeddings=similar_chunk_embeddings, 
        threshold=.25
    )
    sorted_query_response = sorted(query_response, key=lambda x: x[1], reverse=True)[:5]
    complete_query_response = ""
    for count, chunk in enumerate(sorted_query_response):
        complete_query_response += f"Source {count+1}\n{10*'-'}\n{chunk[0]}\n\n"

    system_prompt = "You are an expert at answering questions. Use information from sources to provide answers."
    prompt = f"{query}\n\n{complete_query_response}"
    print(f"{10*'-'}\n{prompt}\n{10*'-'}\n")
    response = openai_text_response(prompt, system_prompt, model_choice="4o")
    print(f"\n\n{response}\n\n")
    