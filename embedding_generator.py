import logging
from typing import List, Dict
import re

from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, embed_dict_list, add_similarity_to_next_dict_item
from genai_toolbox.chunk_and_embed.chunking_functions import convert_utterance_speaker_to_speakers, consolidate_similar_utterances, add_metadata_to_chunks, format_speakers_in_utterances, milliseconds_to_minutes_in_utterances, rename_start_end_to_ms
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from genai_toolbox.helper_functions.datetime_helpers import convert_date_format

def process_entry(
    entry: Dict,
    consolidation_threshold: float = 0.35
) -> List[Dict]:
    feed_title = entry['feed_title']
    episode_title = entry['title']
    episode_date = convert_date_format(entry['published'])
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
    consolidated_similar_utterances = consolidate_similar_utterances(filtered_utterances, similarity_threshold=consolidation_threshold)
    consolidated_embeddings = embed_dict_list(
        embedding_function=create_openai_embedding,
        chunk_dicts=consolidated_similar_utterances, 
        key_to_embed="text",
        model_choice="text-embedding-3-large"
    )
    additional_metadata = {
        "title": f"{feed_title} - {episode_date}: {episode_title}"
    }
    titled_utterances = add_metadata_to_chunks(
        chunks=consolidated_embeddings,
        additional_metadata=additional_metadata
    )
    formatted_utterances = format_speakers_in_utterances(titled_utterances)
    minutes_utterances = milliseconds_to_minutes_in_utterances(formatted_utterances)
    renamed_utterances = rename_start_end_to_ms(minutes_utterances)
    
    for utterance in renamed_utterances:
        start = utterance['start_ms']
        feed_regex = re.sub(r'[^a-zA-Z0-9\s]', '', feed_title)
        episode_regex = re.sub(r'[^a-zA-Z0-9\s]', '', episode_title)
        utterance['id'] = f"{start} {feed_regex} {episode_regex}".replace(' ', '-')
    
    return renamed_utterances
    
def load_existing_embeddings(
    embedding_config: Dict
) -> List[Dict]:
    try:
        return retrieve_file(
            file=embedding_config['existing_embeddings_file'], 
            dir_name=embedding_config['existing_embeddings_dir']
        )
    except FileNotFoundError:
        logging.info("No existing aggregated chunked embeddings found. Creating new file.")
        return []

def generate_embeddings(
    embedding_config: Dict
) -> List[Dict]:
    feed_dict = retrieve_file(
        file=embedding_config['input_podcast_file'],
        dir_name=embedding_config['input_podcast_dir']
    )

    aggregated_chunked_embeddings = [
        embedding for entry in feed_dict
        for embedding in process_entry(entry)
    ]
    
    all_aggregated_chunked_embeddings = load_existing_embeddings(embedding_config)
    if all_aggregated_chunked_embeddings:
        all_aggregated_chunked_embeddings.extend(aggregated_chunked_embeddings)
    else:
        all_aggregated_chunked_embeddings = aggregated_chunked_embeddings
    
    write_to_file(
        content=all_aggregated_chunked_embeddings,
        file=embedding_config['output_file_name'],
        output_dir_name=embedding_config['output_dir_name']
    )

    return all_aggregated_chunked_embeddings