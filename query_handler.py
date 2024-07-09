from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response, anthropic_text_response
from genai_toolbox.helper_functions.string_helpers import retrieve_file
import json

def handle_query(
    question: str,
    similar_chunks_file: str,
    dir_name: str
) -> dict:

    similar_chunks = retrieve_file(
        file=similar_chunks_file, 
        dir_name=dir_name
    )

    llm_system_prompt = """
    Use numbered references to cite the sources that are given to you.
    Use the numbers to identify sources at the bottom of answer with their titles.
    Each timestamp is its own reference (e.g. [1] Title at 01:00). 
    Do not refer to the source material in your text, only in your number citations.
    Give a thorough, detailed, and comprehensive answer.
    """

    source_template = "Title: {title} at {start_mins}\nText: {text}"
    template_args = {
        "title": "title",
        "text": "text",
        "start_mins": "start_mins"
    }

    response = llm_response_with_query(
        similar_chunks=similar_chunks,
        question=question,
        llm_system_prompt=llm_system_prompt,
        source_template=source_template,
        template_args=template_args,
        llm_function=groq_text_response,
        llm_model_choice="llama3-70b",
    )

    return response

if __name__ == "__main__":
    from sql_operations import cosine_similarity_search
    from cloud_sql_gcp.config.gcp_sql_config import load_config
    from genai_toolbox.helper_functions.string_helpers import write_to_file
    
    table_name = 'vector_table'
    config = load_config()
    query_embedding = create_openai_embedding(
        text="What's the latest with Adobe and Figma?",
        model_choice="text-embedding-3-large"
    )

    similar_chunks = cosine_similarity_search(
        config=config,
        table_name=table_name,
        query_embedding=query_embedding,
        limit=5
    )

    similar_chunks_file = "similar_chunks.json"
    write_to_file(
        file=similar_chunks_file,
        output_dir_name="tmp",
        content=similar_chunks
    )

    similar_chunks = retrieve_file(
        file=similar_chunks_file,
        dir_name="tmp"
    )
    for chunk in similar_chunks:
        print(chunk['similarity'])

    response = handle_query(
        question="What were the major trends of 2023?",
        similar_chunks_file=similar_chunks_file,
        dir_name="tmp"
    )

    print(response['llm_response'])



