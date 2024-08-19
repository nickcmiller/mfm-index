import os
import logging
from typing import List, Dict, Any, Generator
import time
from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.chunk_and_embed.llms_with_queries import stream_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import openai_text_response, fallback_text_response
from gcp_postgres_pgvector.config.gcp_sql_config import load_config
from gcp_postgres_pgvector.databases.connection import get_db_engine
from gcp_postgres_pgvector.databases.operations import read_similar_rows

logger = logging.getLogger(__name__)
config = load_config()

def cosine_similarity_search(
    table_name, 
    query_embedding, 
    read_limit=15,
    similarity_threshold=0.30,
    config=config
) -> List[Dict[str, Any]]:
    """
        Perform a cosine similarity search on the specified table.

        This function connects to the database and retrieves rows that are similar to the provided query embedding.
        It filters the results based on a similarity threshold and limits the number of returned rows. The function
        also logs the process, including the number of rows returned and the filtering criteria applied.

        Args:
            table_name (str): The name of the table to search in.
            query_embedding (list): The query embedding to compare against.
            read_limit (int): The maximum number of similar rows to return (default is 15).
            similarity_threshold (float): The minimum similarity score for a row to be included in the results (default is 0.30).
            filter_limit (int): The maximum number of rows to return after filtering based on similarity (default is 10).
            max_similarity_delta (float): The maximum allowable difference from the highest similarity score for filtering (default is 0.075).
            config (dict): A dictionary containing the database configuration parameters.

        Returns:
            list: A list of dictionaries containing the most similar rows that meet the filtering criteria.

        Raises:
            Exception: If there's an error during the database search operation or if the connection to the database fails.
    """
    with get_db_engine(config) as engine:
        logger.info(f"Connected to database: {engine}")
        try:
            start_time = time.time()
            similar_rows = read_similar_rows(
                engine, 
                table_name, 
                query_embedding,
                included_columns=['id', 'text', 'title', 'start_mins', 'youtube_link'],
                limit=read_limit,
                similarity_threshold=similarity_threshold
            )
            time_taken = time.time() - start_time
            logger.info(f"Returned {len(similar_rows)} rows from table '{table_name}' above threshold {similarity_threshold} in {time_taken:.2f} seconds")

            if not similar_rows:
                logger.warning(f"No similar rows found for the query embedding.")
                return [] 

            return similar_rows
        except Exception as e:
            logger.error(f"Failed to perform cosine similarity search: {e}", exc_info=True)
            raise

def retrieve_similar_chunks(
    table_name: str, 
    question: str,
    chat_messages: List[Dict] = [],
    filter_limit=10,
    max_similarity_delta=0.075,
) -> List[Dict]:
    try:
        vectordb_question = _generate_vectordb_question(
            question=question,
            chat_messages=chat_messages
        )
        query_embedding = create_openai_embedding(
            text=vectordb_question, 
            model_choice="text-embedding-3-large"
        )
    except Exception as e:
        logger.error(f"Failed to create query embedding: {e}", exc_info=True)
        raise

    filtered_rows = []
    
    try:
        similar_rows = cosine_similarity_search(
            table_name=table_name, 
            query_embedding=query_embedding,
        )

        max_similarity = max(row['similarity'] for row in similar_rows)
        filtered_rows = [row for row in similar_rows if max_similarity - row['similarity'] <= max_similarity_delta]
        filtered_rows = filtered_rows[:filter_limit]
        logger.info(f"Filtered down to {len(filtered_rows)} rows within {max_similarity_delta} of the highest similarity score: {max_similarity}")

        for row in filtered_rows:
            logger.info(f"Similarity: {row['similarity']} - {row['title']} {row['start_mins']}")
    except Exception as e:
        logger.error(f"Failed to perform cosine similarity search: {e}", exc_info=True)
        raise

    return filtered_rows

def _generate_vectordb_question(
    chat_messages: List[Dict], 
    question: str
) -> str:
    vectordb_prompt = f"""
        Request: {question}\n\nBased on this request, what request should I make to my vector database?
        Use prior messages to establish the intent and context of the question. 
        Include any relevant topics, themes, or individuals mentioned in the chat history. 
        Significantly lengthen the request and include as many contextual details as possible to enhance the relevance of the query.
        Only return the request. Don't preface it or provide an introductory message.
    """
    vectordb_system_instructions = "You expand on questions asked to a vector database containing chunks of transcripts. You add sub-questions and contextual details to make the query more specific and relevant to the chat history."

    fallback_model_order = [
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        }, 
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        }
    ]  

    start_time = time.time()
    vectordb_question = fallback_text_response(
        prompt=vectordb_prompt,
        model_order=fallback_model_order,
        history_messages=chat_messages,
        system_instructions=vectordb_system_instructions,
    )
    time_taken = time.time() - start_time
    logger.info(f"Vector database question: {vectordb_question}\nModel: {fallback_model_order[0]['model']}\n Vectordb time taken: {time_taken:.2f} seconds\n")
    return vectordb_question
    
def stream_question_response(
    question: str, 
    similar_chunks: List[Dict]
) -> Generator[str, None, None]:

    llm_system_prompt = """
    Use numbered references to cite sources with their titles.
    Record each reference in a markdown numbered list at the bottom with a timestamp and a link to the source.
    When a timestamp is reused, use the same number.

    Example 1: 
    '''
    Sentence using first source.[1] Sentence using second and third sources.[2][3]

   **References:**
    1. Source 1 Title at [0:32](https://youtube.com/watch?v=dQw4w9WgXcQ&t=32)
    2. Source 2 Title at [8:47](https://youtube.com/watch?v=oHg5SJYRHA0&t=527)
    3. Source 3 Title at [13:36](https://youtube.com/watch?v=xvFZjo5PgG0&t=816)
    '''

    Example 2: 
    '''
    Here's a list:
    - Sentence using first source.[1]
    - Sentence using second and third sources.[2][3]
    - Sentence using first and third sources.[1][3]

   **References:**
    1. Source 1 Title at [0:32](https://youtube.com/watch?v=dQw4w9WgXcQ&t=32)
    2. Source 2 Title at [8:47](https://youtube.com/watch?v=oHg5SJYRHA0&t=527)
    3. Source 3 Title at [13:36](https://youtube.com/watch?v=xvFZjo5PgG0&t=816)
    '''

    Provide a thorough, detailed, and comprehensive answer.
    If the sources are not relevant to the question, say that you couldn't find any relevant information.
    Before using the sources, ensure that sources are using the same names as the ones in the chat history.
    """

    source_template = "Title: {title} at [{start_mins}]({youtube_link})\nText: {text}"
    template_args = {"title": "title", "text": "text", "start_mins": "start_mins", "youtube_link": "youtube_link"}

    fallback_model_order = [
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        },
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        },
    ]


    print(f"\n\nQuestion: {question}\nModel: {fallback_model_order[0]['model']}\n\n")
    return stream_response_with_query(
        similar_chunks=similar_chunks,
        question=question,
        llm_system_prompt=llm_system_prompt,
        source_template=source_template,
        template_args=template_args,
        llm_model_order=fallback_model_order
    )

def question_with_chat_state(
    question: str, 
    chat_state: List[Dict],
    table_name: str
) -> Generator[str, None, None]:    

    similar_chunks = retrieve_similar_chunks(
        table_name=table_name, 
        question=question,
        chat_messages=chat_state
    )

    revised_question = _generate_revised_question(chat_state, question)

    return stream_question_response(
        question=revised_question, 
        similar_chunks=similar_chunks
    )

def _generate_revised_question(
    chat_messages: List[Dict], 
    question: str
) -> str:
    revision_prompt = f"""
        Question: {question}
        When possible, rewrite the question using <chat history> to identify the intent of the question, the people referenced by the question, and ideas / topics / themes targeted by the question in <chat history>.
        If the <chat history> does not contain any information about the people, ideas, or topics relevant to the question, then do not make any assumptions.
        Only return the request. Don't preface it or provide an introductory message.
    """
    revision_system_instructions = "You are an assistant that concisely and carefully rewrites questions. The less than (<) and greater than (>) signs are telling you to refer to the chat history. Don't use < or > in your response."

    fallback_model_order = [
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        }, 
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        }
    ]  
    start_time = time.time()
    revised_question = fallback_text_response(
        prompt=revision_prompt,
        model_order=fallback_model_order,
        history_messages=chat_messages,
        system_instructions=revision_system_instructions,
    )
    time_taken = time.time() - start_time
    logger.info(f"Revised question: {revised_question}\nModel: {fallback_model_order[0]['model']}\nRevision time taken: {time_taken:.2f} seconds\n")
    return revised_question


if __name__ == "__main__":
    import json
    
    table_name = 'vector_table'
    question = "What are businesses I can start in regulated industries?"

    chat_state = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that answers questions about encryption in regulation."
        },
        {
            "role": "user", 
            "content": "What is the relevance of encryption in regulation?"
        },
        {
            "role": "assistant",
            "content": """
    Encryption plays a crucial role in regulation, particularly in the context of balancing security, privacy, and control over information. Historically, encryption has been a contentious issue in regulatory frameworks, as evidenced by the export laws in the 1990s that restricted the export of encryption algorithms above a certain cryptographic strength. These regulations were aimed at preventing the misuse of strong encryption by adversaries but were often seen as overly restrictive and counterproductive by technologists and privacy advocates [1].

    The relevance of encryption in regulation can be broken down into several key aspects:

    1. **National Security and Law Enforcement**: Governments often regulate encryption to ensure that law enforcement and intelligence agencies can access communications when necessary. This is driven by the fear that strong encryption could be used by malicious actors to conceal their activities, making it difficult for authorities to intercept and decipher communications [2]. However, this creates a tension between the need for security and the right to privacy.

    2. **Privacy and Civil Liberties**: Encryption is fundamental to protecting the privacy of individuals and organizations. It ensures that sensitive information, such as personal communications, financial transactions, and proprietary business data, remains confidential and secure from unauthorized access. The debate around encryption regulation often centers on finding a balance between enabling government access for security purposes and protecting individual privacy rights [3].

    3. **Technological Innovation and Economic Impact**: Overly restrictive encryption regulations can stifle innovation and economic growth. For example, the limitations on exporting strong encryption in the 1990s were seen as hindering the development and global competitiveness of the tech industry. Today, encryption is a critical component of many technologies, from secure messaging apps to online banking, and regulations need to consider the potential impact on technological advancement and economic activity [4].

    4. **Global Standards and Interoperability**: Encryption regulations can vary significantly between countries, leading to challenges in creating globally interoperable systems. Companies operating internationally must navigate a complex landscape of regulatory requirements, which can complicate the development and deployment of secure technologies. Harmonizing encryption standards and regulations across borders is essential for ensuring that security measures are effective and universally applicable [5].

    5. **Public Perception and Trust**: The regulation of encryption also affects public trust in technology and government. If people believe that their communications and data are not secure, they may be less likely to use digital services, which can have broader implications for digital adoption and trust in institutions. Conversely, transparent and balanced encryption regulations can enhance public confidence in the security and privacy of digital services [6].

    In summary, encryption is a pivotal element in the regulatory landscape, influencing national security, privacy, technological innovation, global interoperability, and public trust. Effective regulation must strike a delicate balance between enabling legitimate government access and protecting the privacy and security of individuals and organizations.
            """
        },
        {
            "role": "user", 
            "content": "Who is Godzilla?"
        },
        {
            "role": "assistant",
            "content": "I'm sorry, but I don't have enough information to answer this question based on the available data."
        }
    ]

    stream_response = question_with_chat_state(
        question=question,
        chat_state=chat_state,
        table_name=table_name,
    )

    chunk_count = 0
    for chunk in stream_response:
        chunk_count += 1
    print(f"\nTotal chunks streamed: {chunk_count}")


