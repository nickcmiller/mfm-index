from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_episode_summary
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_utterances, replace_speakers_in_assemblyai_utterances
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, embed_dict_list, add_similarity_to_next_dict_item
from genai_toolbox.chunk_and_embed.chunking_functions import convert_utterance_speaker_to_speakers, consolidate_similar_utterances, add_metadata_to_chunks, format_speakers_in_utterances, milliseconds_to_minutes_in_utterances
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.text_prompting.model_calls import anthropic_text_response, groq_text_response, openai_text_response
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from genai_toolbox.helper_functions.datetime_helpers import convert_date_format

import json
import os
import logging
from typing import List, Dict, Optional
import traceback

import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm

async def download_and_transcribe_multiple_episodes_by_date(
    feed_url: str,
    start_date: str,
    end_date: str,
    audio_dir_name: Optional[str] = None,
    utterances_dir_name: Optional[str] = None,
    utterances_replaced_dir_name: Optional[str] = None,
    max_concurrent_tasks: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, str]]:
    feed_entries = return_entries_by_date(feed_url, start_date, end_date)
    logging.info(f"Downloading {len(feed_entries)} episodes")

    semaphore = Semaphore(max_concurrent_tasks)

    async def process_entry(entry: Dict[str, str], pbar: tqdm) -> Dict[str, str]:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    pbar.set_description(f"Processing {entry['title'][:30]}...")
                    
                    entry['audio_file_path'] = await asyncio.to_thread(
                        download_podcast_audio, 
                        entry['url'], 
                        entry['title'], 
                        download_dir_name=audio_dir_name
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
                        output_dir_name=utterances_dir_name
                    )
                    pbar.set_postfix({"stage": "utterances generated"})
                    
                    entry['replaced_dict'] = await asyncio.to_thread(
                        replace_speakers_in_assemblyai_utterances, 
                        entry['utterances_dict'], 
                        entry['audio_summary'],
                        output_dir_name=utterances_replaced_dir_name
                    )
                    pbar.set_postfix({"stage": "utterances replaced"})

                    if os.path.exists(entry['audio_file_path']):
                        os.remove(entry['audio_file_path'])
                        logging.info(f"Removed audio file: {entry['audio_file_path']}")
                    else:
                        logging.warning(f"Audio file not found for cleanup: {entry['audio_file_path']}")
                    
                    pbar.update(1)
                    return entry
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Error processing entry {entry['title']}, attempt {attempt + 1}: {str(e)}")
                        logging.debug(f"Traceback: {traceback.format_exc()}")
                        pbar.set_postfix({"stage": f"retry {attempt + 1}/{max_retries}"})
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        logging.error(f"Failed to process entry {entry['title']} after {max_retries} attempts: {str(e)}")
                        logging.debug(f"Traceback: {traceback.format_exc()}")
                        pbar.set_postfix({"stage": "failed"})
                        pbar.update(1)
            return None

    with tqdm(total=len(feed_entries), desc="Processing episodes") as pbar:
        tasks = [process_entry(entry, pbar) for entry in feed_entries]
        updated_entries = await asyncio.gather(*tasks)

    successful_entries = [entry for entry in updated_entries if entry is not None]
    logging.info(f"Successfully processed {len(successful_entries)} out of {len(feed_entries)} episodes")
    return successful_entries

async def main():
    feed_url = "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ"
    start_date_input = "March 1, 2024"
    end_date_input = "April 1, 2024"
    audio_dir_name = "tmp_audio"

    if False:
        updated_entries = await download_and_transcribe_multiple_episodes_by_date(
            feed_url=feed_url,
            start_date=start_date_input,
            end_date=end_date_input,
            audio_dir_name=audio_dir_name,
        )
        existing_entries = retrieve_file(
            file="podcast_feed.json",
            dir_name="tmp"
        )
        all_entries = existing_entries + updated_entries
        write_to_file(
            content=all_entries,
            file="podcast_feed.json",
            output_dir_name="tmp"
        )
        write_to_file(
            content=updated_entries,
            file="updated_entries.json",
            output_dir_name="tmp"
        )
    else:
        all_entries = retrieve_file("podcast_feed.json", dir_name="tmp")
        updated_entries = retrieve_file("updated_entries.json", dir_name="tmp")

if __name__ == "__main__":
    asyncio.run(main())

    feed_dict = retrieve_file(
        file="updated_entries.json", 
        dir_name="tmp"
    )

    aggregated_chunked_embeddings = []
    if False:
        for entry in feed_dict:
            feed_title = entry['feed_title']
            episode_title = entry['title']
            episode_date=convert_date_format(entry['published'])
            utterances = entry['replaced_dict']['transcribed_utterances']
            speakermod_utterances = convert_utterance_speaker_to_speakers(utterances)
            embedded_utterances = embed_dict_list(
                embedding_function=create_openai_embedding,
                chunk_dicts=speakermod_utterances, 
                key_to_embed="text",
                model_choice="text-embedding-3-large"
            )
            similar_utterances = add_similarity_to_next_dict_item(embedded_utterances)
            filtered_utterances = [
                {k: v for k, v in utterance.items() if k != 'embedding'}
                for utterance in similar_utterances
            ]
            consolidated_similar_utterances = consolidate_similar_utterances(filtered_utterances, similarity_threshold=0.35)
            consolidated_embeddings = embed_dict_list(
                embedding_function=create_openai_embedding,
                chunk_dicts=consolidated_similar_utterances, 
                key_to_embed="text",
                model_choice="text-embedding-3-large"
            )
            additional_metadata = {
                "title": f" {feed_title} - {episode_date}: {episode_title}"
            }
            titled_embeddings = add_metadata_to_chunks(
                chunks=consolidated_embeddings,
                additional_metadata=additional_metadata
            )
            formatted_embeddings = format_speakers_in_utterances(titled_embeddings)
            milliseconds_embeddings = milliseconds_to_minutes_in_utterances(formatted_embeddings)
            aggregated_chunked_embeddings.extend(milliseconds_embeddings)
        existing_aggregated_chunked_embeddings = retrieve_file(
            file="aggregated_chunked_embeddings.json", 
            dir_name="tmp"
        )
        all_aggregated_chunked_embeddings = existing_aggregated_chunked_embeddings + aggregated_chunked_embeddings
        write_to_file(
            content=all_aggregated_chunked_embeddings,
            file="aggregated_chunked_embeddings.json",
            output_dir_name="tmp"
        )
    else:
        aggregated_chunked_embeddings = retrieve_file(
            file="aggregated_chunked_embeddings.json", 
            dir_name="tmp"
        )
question = "Why is NVIDIA's stock rising?"
llm_system_prompt = f"""
Use numbered references to cite the sources that are given to you. 
Each timestamp is its own reference (e.g. [1] Title at 01:00). 
Do not refer to the source material in your text, only in your number citations
Give a detailed answer.
"""
source_template="Title: {title} at {start_time}\nText: {text}"
template_args={
    "title": "title",
    "text": "text",
    "start_time": "start_time",
}

if True:
    response = llm_response_with_query(
        question=question,
        chunks_with_embeddings=aggregated_chunked_embeddings,
        embedding_function=create_openai_embedding,
        query_model="text-embedding-3-large",
        threshold=0.35,
        max_query_chunks=5,
        llm_function=groq_text_response,
        llm_model_choice="llama3-70b",
        source_template=source_template,
        template_args=template_args,
    )

    # print(response)
    print(len(response['query_response']))
    print(json.dumps(response['query_response'], indent=4))
    print(f"\n\nQuestion: {question}\n\n")
    print(f"Response: {response['llm_response']}\n\n")