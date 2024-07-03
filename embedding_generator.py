from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, embed_dict_list, add_similarity_to_next_dict_item
from genai_toolbox.chunk_and_embed.chunking_functions import convert_utterance_speaker_to_speakers, consolidate_similar_utterances, add_metadata_to_chunks, format_speakers_in_utterances, milliseconds_to_minutes_in_utterances
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from genai_toolbox.helper_functions.datetime_helpers import convert_date_format

def generate_embeddings():
    feed_dict = retrieve_file("updated_entries.json", dir_name="tmp")
    aggregated_chunked_embeddings = []
    
    for entry in feed_dict:
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
        consolidated_similar_utterances = consolidate_similar_utterances(filtered_utterances, similarity_threshold=0.35)
        consolidated_embeddings = embed_dict_list(
            embedding_function=create_openai_embedding,
            chunk_dicts=consolidated_similar_utterances, 
            key_to_embed="text",
            model_choice="text-embedding-3-large"
        )
        additional_metadata = {
            "title": f"{feed_title} - {episode_date}: {episode_title}"
        }
        titled_embeddings = add_metadata_to_chunks(
            chunks=consolidated_embeddings,
            additional_metadata=additional_metadata
        )
        formatted_embeddings = format_speakers_in_utterances(titled_embeddings)
        minutes_embeddings = milliseconds_to_minutes_in_utterances(formatted_embeddings)
        aggregated_chunked_embeddings.extend(minutes_embeddings)
    
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
    
    return all_aggregated_chunked_embeddings

if __name__ == "__main__":
    generate_embeddings()