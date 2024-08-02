import os
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding
from genai_toolbox.text_prompting.model_calls import openai_text_response, fallback_text_response
from gcp_postgres_pgvector.config.gcp_sql_config import load_config
from gcp_postgres_pgvector.databases.connection import get_db_engine
from gcp_postgres_pgvector.databases.operations import read_similar_rows
from gcp_postgres_pgvector.utils.logging import setup_logging

logger = setup_logging()
config = load_config()
# Log the config before attempting to connect
logger.info(f"Config: {config}")

def cosine_similarity_search(
    table_name, 
    query_embedding, 
    read_limit=15,
    similarity_threshold=0.30,
    filter_limit=10,
    max_similarity_delta=0.075,
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
            similar_rows = read_similar_rows(
                engine, 
                table_name, 
                query_embedding,
                included_columns=['id', 'text', 'title', 'start_mins', 'youtube_link'],
                limit=read_limit,
                similarity_threshold=similarity_threshold
            )
            logger.info(f"Returned {len(similar_rows)} rows from table '{table_name}' above threshold {similarity_threshold}")

            max_similarity = max(row['similarity'] for row in similar_rows)
            filtered_rows = [row for row in similar_rows if max_similarity - row['similarity'] <= max_similarity_delta]
            filtered_rows = filtered_rows[:filter_limit]
            logger.info(f"Filtered down to {len(filtered_rows)} rows within {max_similarity_delta} of the highest similarity score: {max_similarity}")

            for row in filtered_rows:
                logger.info(f"Similarity: {row['similarity']} - {row['title']} {row['start_mins']}")
            
            return filtered_rows
        except Exception as e:
            logger.error(f"Failed to perform cosine similarity search: {e}", exc_info=True)
            raise

def single_question(
    question: str, 
    similar_chunks: List[Dict]
) -> dict:
    if not similar_chunks:
        return {
            "llm_response": "I'm sorry, but I don't have enough information to answer this question based on the available data.",
            "similar_chunks": []
        }

    llm_system_prompt = """
    Use numbered references to cite sources with their titles.
    Record each reference in a markdown numbered list at the bottom with a timestamp and a link to the source.
    Use the same number when a citation is reused.

    Example: 
    ```
    Sentence using first source.[1] Sentence using second and third sources.[2][3]

   **References:**
    1. Source 1 Title at [0:32](https://youtube.com/watch?v=dQw4w9WgXcQ&t=32)
    2. Source 2 Title at [8:47](https://youtube.com/watch?v=oHg5SJYRHA0&t=527)
    3. Source 3 Title at [13:36](https://youtube.com/watch?v=xvFZjo5PgG0&t=816)
    ```
    
    Provide a thorough, detailed, and comprehensive answer in paragraph form.
    """

    source_template = "Title: {title} at [{start_mins}]({youtube_link})\nText: {text}"
    template_args = {"title": "title", "text": "text", "start_mins": "start_mins", "youtube_link": "youtube_link"}

    return llm_response_with_query(
        similar_chunks=similar_chunks,
        question=question,
        llm_system_prompt=llm_system_prompt,
        source_template=source_template,
        template_args=template_args,
        llm_model_order=[
            {
                "provider": "openai", 
                "model": "4o-mini"
            },
            {
                "provider": "groq", 
                "model": "llama3.1-70b"
            },
            {
                "provider": "anthropic", 
                "model": "sonnet"
            },
        ],
    )

def question_with_chat_state(
    question: str, 
    chat_state: List[Dict], 
    table_name: str
) -> Dict:
    """
        Process a question with chat context and retrieve relevant information from a vector database.

        This function takes a user's question, the current chat state, and a table name for the vector database.
        It then performs the following steps:
        1. Generates a revised question based on the chat context using a language model.
        2. Creates an embedding for the revised question.
        3. Performs a cosine similarity search in the vector database to find relevant chunks of information.
        4. Generates a response using the retrieved information and the revised question.

        Args:
            question (str): The user's current question.
            chat_state (List[Dict]): A list of dictionaries representing the chat history.
                Each dictionary should have 'role' and 'content' keys.
            table_name (str): The name of the table in the vector database to search.

        Returns:
            Dict: A dictionary containing the response from the language model and similar chunks of information.
                The structure is determined by the `single_question` function.

        Note:
            This function relies on several helper functions and external services, including:
            - openai_text_response: For generating the revised question.
            - create_openai_embedding: For creating embeddings.
            - cosine_similarity_search: For searching the vector database.
            - single_question: For generating the final response.
    """
    prompt = f"""
    Question: {question}\n\nBased on this question, what question should I ask my vector database? 
    Only use prior messages if they are relevant to the question.
    Example outputs:
    ```
    What is the relevance of encryption in regulation?
    ```
    ```
    Who is Harry Potter?
    ```
    ```
    How can I learn to swim?
    ```
    """
    question_system_instructions = "Return only the question to be asked. No formatting, just the question."

    chat_messages = chat_state[-5:][::-1] + [{"role": "user", "content": question}, {"role": "assistant", "content": "I will follow the instructions."}]
    logger.info(f"Chat messages: {chat_messages}")
    revised_question = openai_text_response(
        prompt=prompt,
        model_choice="4o-mini",
        history_messages=chat_messages,
        system_instructions=question_system_instructions,
    )
    logger.info(f"\n\nRevised question: {revised_question}\n\n")

    query_embedding = create_openai_embedding(text=revised_question, model_choice="text-embedding-3-large")
    similar_chunks = cosine_similarity_search(
        table_name=table_name, 
        query_embedding=query_embedding,
    )

    return single_question(question=revised_question, similar_chunks=similar_chunks)
    

if __name__ == "__main__":
    import json
    
    table_name = 'vector_table'
    question = "How would I start a business from scratch?"

    chat_state = [
    #     {
    #         "role": "user", 
    #         "content": "What is the relevance of encryption in regulation?"
    #     },
    #     {
    #         "role": "assistant",
    #         "content": """
    # Encryption plays a crucial role in regulation, particularly in the context of balancing security, privacy, and control over information. Historically, encryption has been a contentious issue in regulatory frameworks, as evidenced by the export laws in the 1990s that restricted the export of encryption algorithms above a certain cryptographic strength. These regulations were aimed at preventing the misuse of strong encryption by adversaries but were often seen as overly restrictive and counterproductive by technologists and privacy advocates [1].

    # The relevance of encryption in regulation can be broken down into several key aspects:

    # 1. **National Security and Law Enforcement**: Governments often regulate encryption to ensure that law enforcement and intelligence agencies can access communications when necessary. This is driven by the fear that strong encryption could be used by malicious actors to conceal their activities, making it difficult for authorities to intercept and decipher communications [2]. However, this creates a tension between the need for security and the right to privacy.

    # 2. **Privacy and Civil Liberties**: Encryption is fundamental to protecting the privacy of individuals and organizations. It ensures that sensitive information, such as personal communications, financial transactions, and proprietary business data, remains confidential and secure from unauthorized access. The debate around encryption regulation often centers on finding a balance between enabling government access for security purposes and protecting individual privacy rights [3].

    # 3. **Technological Innovation and Economic Impact**: Overly restrictive encryption regulations can stifle innovation and economic growth. For example, the limitations on exporting strong encryption in the 1990s were seen as hindering the development and global competitiveness of the tech industry. Today, encryption is a critical component of many technologies, from secure messaging apps to online banking, and regulations need to consider the potential impact on technological advancement and economic activity [4].

    # 4. **Global Standards and Interoperability**: Encryption regulations can vary significantly between countries, leading to challenges in creating globally interoperable systems. Companies operating internationally must navigate a complex landscape of regulatory requirements, which can complicate the development and deployment of secure technologies. Harmonizing encryption standards and regulations across borders is essential for ensuring that security measures are effective and universally applicable [5].

    # 5. **Public Perception and Trust**: The regulation of encryption also affects public trust in technology and government. If people believe that their communications and data are not secure, they may be less likely to use digital services, which can have broader implications for digital adoption and trust in institutions. Conversely, transparent and balanced encryption regulations can enhance public confidence in the security and privacy of digital services [6].

    # In summary, encryption is a pivotal element in the regulatory landscape, influencing national security, privacy, technological innovation, global interoperability, and public trust. Effective regulation must strike a delicate balance between enabling legitimate government access and protecting the privacy and security of individuals and organizations.
    #         """
    #    }
        {
            "role": "user", 
            "content": "Who is Godzilla?"
        },
        {
            "role": "assistant",
            "content": "I'm sorry, but I don't have enough information to answer this question based on the available data."
        }
    ]

    response = question_with_chat_state(
        question=question,
        chat_state=chat_state,
        table_name=table_name,
    )

    # print(json.dumps(response['similar_chunks'], indent=4))
    print(response['llm_response'])
    # print(response['similar_chunks'])



