from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_audio_summary
from genai_toolbox.helper_functions.string_helpers import write_string_to_file, retrieve_string_from_file, evaluate_and_clean_valid_response
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_transcript, replace_assemblyai_speakers
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.text_prompting.model_calls import openai_text_response

import json
import logging

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

def split_text_string(text, separator):
    chunks = text.split(separator)
    return chunks
chunks = split_text_string(replaced_transcript, "\n\n")

client = openai_client()

def create_semantic_indices(
    text: str
) -> list[dict]:
    system_instructions = """
    Given a conversation, produce a JSON list of dictionaries that achieve the following: 
    - Analyze the flow of the conversation
    - Identify shifts in topic or context
    - Group related messages into chunks based on semantic similarity
    - Preserve the chronological order of messages within each chunk
    - Provide start and end indices for each proposed chunk in the original text

    Examples of properly formatted JSON output:
    ```
    [
        {
            "topic": "Topic 1", 
            "start_index": 0,
            "end_index": 237
        },
        {
            "topic": "Topic 2",
            "start_index": 238, 
            "end_index": 1522
        },
        {
            "topic": "Topic 3",
            "start_index": 1523,
            "end_index": 2879
        },
        ...
    ]
    ```
    """

    prompt = f"""
    Conversation: {text}
    """
    count = 0
    while count < 5:    
        try:
            response = openai_text_response(prompt, system_instructions=system_instructions)
            logging.info(f"Response: {response}")
            valid_response = evaluate_and_clean_valid_response(response, list)
            if all(isinstance(item, dict) for item in valid_response):
                return valid_response
            else:
                raise ValueError(f"Response items are not of type {expected_type}")
        except Exception as e:
            logging.error(f"Error: {e}")
            count += 1
    logging.error(f"Failed to get valid response after {count} attempts")
    return []

if True:
    semantic_indices = create_semantic_indices(replaced_transcript)
    write_string_to_file("semantic_indices.txt", json.dumps(semantic_indices, indent=4))
else:
    semantic_indices = retrieve_string_from_file("semantic_indices.txt")

print(json.dumps(semantic_indices, indent=4))    
print(type(semantic_indices))


def chunk_document(
    document: str, 
    semantic_chunks: list[dict]
) -> list[dict]:
    chunks = []
    for chunk_info in semantic_chunks:
        start_index = chunk_info['start_index']
        end_index = chunk_info['end_index']
        chunk_text = document[start_index:end_index]
        chunks.append({
            'topic': chunk_info['topic'],
            'text': chunk_text
        })
    return chunks

chunks = chunk_document(replaced_transcript, semantic_indices)
print(json.dumps(chunks, indent=4))



def create_embedding(chunk):
    response = client.embeddings.create(
        input=chunk, 
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


if False:
    embeddings = [create_embedding(chunk) for chunk in chunks]
    write_string_to_file("embeddings.txt", json.dumps(embeddings, indent=4))
else:
    embeddings = retrieve_string_from_file("embeddings.txt")

