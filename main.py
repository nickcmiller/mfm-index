from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_audio_summary
from genai_toolbox.helper_functions.string_helpers import write_string_to_file, retrieve_string_from_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_transcript, replace_assemblyai_speakers
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.text_prompting.model_calls import openai_text_response

import json
import logging
import numpy as np
from typing import Callable

# Example usage
start_date_input = "June 4, 2024"
end_date_input = "June 6, 2024"

feed_url = "https://feeds.megaphone.fm/HS2300184645"
feed_entries = return_entries_by_date(feed_url, start_date_input, end_date_input)
write_string_to_file("mfm_feed.txt", json.dumps(feed_entries, indent=4))

"""
Sample Filter Entry:
[
    {
        "entry_id": "cdf26398-2ca3-11ef-a5ee-975153545b62",
        "title": "EXCLUSIVE: $3B Founder Reveals His Next Big Idea",
        "published": "Mon, 17 Jun 2024 14:00:00 -0000",
        "summary": "Episode 597: Sam Parr ( https://twitter.com/theSamParr ) talks to Brett Adcock ( https://x.com/adcock_brett ) about his next big idea, his checklist for entrepreneurs, and his framework for learning new things and moving fast.\u00a0\n\n\u2014\nShow Notes:\n(0:00) Solving school shootings\n(3:15) Cold calling NASA\n(6:14) Spotting the mega-trend\n(8:37) \"Thinking big is easier\"\n(12:42) Brett's philosophy on company names\n(16:22) Brett's ideas note app: genetics, super-sonic travel, synthetic foods\n(19:45) \"I just want to win\"\n(21:46) Brett's checklist for entrepreneurs\n(25:17) Being fast in hardware\n(30:15) Brett's framework for learning new things\n(33:00) Who does Brett admire\n\n\u2014\nLinks:\n\u2022 [Steal This] Get our proven writing frameworks that have made us millions https://clickhubspot.com/copy\n\u2022 Brett Adcock - https://www.brettadcock.com/\n\u2022 Cover - https://www.cover.ai/\n\u2022 Figure - https://figure.ai/\n\n\u2014\nCheck Out Shaan's Stuff:\nNeed to hire? You should use the same service Shaan uses to hire developers, designers, & Virtual Assistants \u2192 it\u2019s called Shepherd (tell \u2018em Shaan sent you): https://bit.ly/SupportShepherd\n\n\u2014\nCheck Out Sam's Stuff:\n\u2022 Hampton - https://www.joinhampton.com/\n\u2022 Ideation Bootcamp - https://www.ideationbootcamp.co/\n\u2022 Copy That - https://copythat.com\n\u2022 Hampton Wealth Survey - https://joinhampton.com/wealth\n\u2022 Sam\u2019s List - http://samslist.co/\n\n\nMy First Million is a HubSpot Original Podcast // Brought to you by The HubSpot Podcast Network // Production by Arie Desormeaux // Editing by Ezra Bakker Trupiano",
        "url": "https://pdst.fm/e/chrt.fm/track/28555/pdrl.fm/2a922f/traffic.megaphone.fm/HS9983733981.mp3?updated=1718638499",
        "feed_summary": "Sam Parr and Shaan Puri brainstorm new business ideas based on trends & opportunities they see in the market. Sometimes they bring on famous guests to brainstorm with them."
    },
]
"""
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

def split_text_string(
    text: str, 
    separator: str
):
    chunks = text.split(separator)
    return chunks
chunks = split_text_string(replaced_transcript, "\n\n")

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

lengthened_chunks = consolidate_short_chunks(chunks, 100)
# for chunk in consolidated_chunks:
#     print(f"{10*'-'}\n{len(chunk)}\n{chunk}\n{10*'-'}\n")
        

client = openai_client()
def create_embedding(
    chunk: str
) -> list[float]:
    response = client.embeddings.create(
        input=chunk, 
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_string_list(
    chunks: list[str]
) -> list[list[float]]:
    return [create_embedding(chunk) for chunk in chunks]

if False:
    lengthened_chunk_embeddings = embed_string_list(consolidated_chunks)
    write_string_to_file("embeddings.txt", json.dumps(lengthened_chunk_embeddings, indent=4))
else:
    lengthened_chunk_embeddings = retrieve_string_from_file("embeddings.txt")
    lengthened_chunk_embeddings = json.loads(lengthened_chunk_embeddings)

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
            print(f"Similarity: {similarity}")
            if similarity > threshold:
                current_chunk.append(chunks[j])
                used_indices.add(j)
                similar_found = True
            else:
                if len(current_chunk) > 1:
                    similar_chunks.append('\n\n'.join(current_chunk))
                    appended_chunk = '\n\n'.join(current_chunk)
                    print(f"{10*'-'}\n{appended_chunk}\n{10*'-'}\n")
                break      
        if similar_found is False:
            similar_chunks.append(chunks[i])
            print(f"{chunks[i]}\n\n")

        used_indices.add(i)
    
    if len(current_chunk) > 0:
        similar_chunks.append('\n\n'.join(current_chunk))
        print(f"{10*'-'}\n{' '.join(current_chunk)}\n{10*'-'}\n")
    
    return similar_chunks

similar_chunks = consolidate_similar_chunks(lengthened_chunks, lengthened_chunk_embeddings, .5)
for chunk in similar_chunks:
    print(f"{len(chunk)}\n--------\n")   



