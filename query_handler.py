from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response, anthropic_text_response
from genai_toolbox.helper_functions.string_helpers import retrieve_file
import json

def handle_query(
    query_config: dict
) -> dict:
    question = query_config['question']
    aggregated_chunked_embeddings = query_config['input_file_name']
    dir_name = query_config['input_dir_name']

    aggregated_chunked_embeddings = retrieve_file(
        file=aggregated_chunked_embeddings, 
        dir_name=dir_name
    )

    llm_system_prompt = """
    Use numbered references to cite the sources that are given to you. 
    Each timestamp is its own reference (e.g. [1] Title at 01:00). 
    Do not refer to the source material in your text, only in your number citations
    Give a detailed answer.
    """

    source_template = "Title: {title} at {start_time}\nText: {text}"
    template_args = {
        "title": "title",
        "text": "text",
        "start_time": "start_time",
    }

    response = llm_response_with_query(
        question=question,
        chunks_with_embeddings=aggregated_chunked_embeddings,
        embedding_function=create_openai_embedding,
        query_model="text-embedding-3-large",
        threshold=0.35,
        max_query_chunks=5,
        llm_function=openai_text_response,
        llm_model_choice="4o",
        source_template=source_template,
        template_args=template_args,
    )

    return response