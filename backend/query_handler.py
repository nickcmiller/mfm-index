from typing import List, Dict, Any
import os

from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response, anthropic_text_response
from gcp_postgres_pgvector.config.gcp_sql_config import load_config
from gcp_postgres_pgvector.databases.connection import get_db_engine
from gcp_postgres_pgvector.databases.operations import read_similar_rows
from gcp_postgres_pgvector.utils.logging import setup_logging

logger = setup_logging()
config = load_config()

def cosine_similarity_search(
    table_name, 
    query_embedding, 
    limit=5,
    config=config
) -> List[Dict[str, Any]]:
    """
        Perform a cosine similarity search on the specified table.

        This function connects to the database and performs a cosine similarity search
        using the provided query embedding. It returns the most similar rows based on
        the cosine similarity between the query embedding and the stored embeddings.

        Args:
            config (dict): A dictionary containing the database configuration parameters.
            table_name (str): The name of the table to search in.
            query_embedding (list): The query embedding to compare against.
            limit (int): The maximum number of similar rows to return.

        Returns:
            list: A list of dictionaries containing the most similar rows and their similarity scores.

        Raises:
            Exception: If there's an error during the database search operation.
    """
    with get_db_engine(config) as engine:
        try:
            similar_rows = read_similar_rows(engine, table_name, query_embedding, limit=limit)
            logger.info(f"Found {len(similar_rows)} similar rows in table '{table_name}'")
            
            for row in similar_rows:
                logger.info(f"Similarity: {row['similarity']}, ID: {row['id']}")
                logger.info(f"Text: {row['text'][:100]}...")
            
            return similar_rows
        except Exception as e:
            logger.error(f"Failed to perform cosine similarity search: {e}", exc_info=True)
            raise

def single_question(
    question: str, 
    similar_chunks: List[Dict]
) -> dict:
    llm_system_prompt = """
    Use numbered references to cite sources with their titles.
    Each timestamp is its own reference in a markdown numbered list at the bottom. 
    ```
   **References:**
    1. Source 1 Title at 0:32
    2. Source 2 Title at 8:47
    3. Source 3 Title at 13:36
    ```
    Use the same number when a citation is reused. 
    Provide a thorough, detailed, and comprehensive answer in paragraph form.
    """

    source_template = "Title: {title} at {start_mins}\nText: {text}"
    template_args = {"title": "title", "text": "text", "start_mins": "start_mins"}

    return llm_response_with_query(
        similar_chunks=similar_chunks,
        question=question,
        llm_system_prompt=llm_system_prompt,
        source_template=source_template,
        template_args=template_args,
        llm_function=anthropic_text_response,
        llm_model_choice="sonnet",
    )

def question_with_chat_state(
    question: str, 
    chat_state: List[Dict], 
    table_name: str
) -> Dict:
    prompt = f"Question: {question}\n\nBased on this question and the prior messages, what question should I ask my vector database?"
    question_system_instructions = "Return only the question to be asked. No formatting, just the question."

    chat_messages = chat_state[-5:][::-1] + [{"role": "system", "content": question_system_instructions}]
    revised_question = groq_text_response(
        prompt=prompt,
        model_choice="llama3-70b",
        history_messages=chat_messages,
        system_instructions=question_system_instructions,
    )

    query_embedding = create_openai_embedding(text=revised_question, model_choice="text-embedding-3-large")
    similar_chunks = cosine_similarity_search(table_name=table_name, query_embedding=query_embedding, limit=5)

    return single_question(question=revised_question, similar_chunks=similar_chunks)
    



