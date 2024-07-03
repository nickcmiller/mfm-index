from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response, anthropic_text_response
from genai_toolbox.helper_functions.string_helpers import retrieve_file
import json

def handle_query(question: str, config: dict):
    aggregated_chunked_embeddings = retrieve_file(
        file="aggregated_chunked_embeddings.json", 
        dir_name="tmp"
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

    print(f"Number of query responses: {len(response['query_response'])}")
    print(json.dumps(response['query_response'], indent=4))
    print(f"\n\nQuestion: {question}\n\n")
    print(f"Response: {response['llm_response']}\n\n")

    return response

if __name__ == "__main__":
    # This is for testing the module directly
    test_question = "Why is NVIDIA's stock rising?"
    test_config = {}  # Add any necessary configuration here
    handle_query(test_question, test_config)