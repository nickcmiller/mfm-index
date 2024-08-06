import os
import logging
from typing import List, Dict, Any, Generator
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

            if not similar_rows:
                logger.warning(f"No similar rows found for the query embedding.")
                return [] 

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
) -> Generator[str, None, None]:
    """
        Generates a response to a single question based on the provided similar chunks of data.

        This function takes a user question and a list of similar chunks retrieved from a database. It constructs a 
        system prompt for a language model (LLM) to generate a comprehensive answer that cites the sources of the 
        information used. The function formats the response to include numbered references to the sources, which 
        are listed at the end of the answer with their respective timestamps and links.

        Args:
            question (str): The question posed by the user that needs to be answered.
            similar_chunks (List[Dict]): A list of dictionaries containing similar data chunks that are relevant to 
                                        the question. Each dictionary should include fields such as 'title', 
                                        'text', 'start_mins', and 'youtube_link'.

        Yields:
            Generator[str, None, None]: A generator that yields the response from the LLM as it is being generated.

        Returns:
            dict: A dictionary containing the LLM's response and the list of similar chunks used to generate the 
                response. If no similar chunks are provided, it returns a default message indicating insufficient 
                information.

        Raises:
            Exception: If there is an error during the generation of the response or if the input data is not 
                    formatted correctly.

        Example:
            question = "What are the key insights from the podcast?"
            similar_chunks = [
                {"title": "Episode 1", "text": "Insight about topic A.", "start_mins": "0:10", "youtube_link": "https://youtube.com/watch?v=example1"},
                {"title": "Episode 2", "text": "Insight about topic B.", "start_mins": "1:20", "youtube_link": "https://youtube.com/watch?v=example2"}
            ]
            response = single_question(question, similar_chunks)
            for chunk in response:
                print(chunk, end='', flush=True)
    """
    
    # # Check if similar_chunks is None or empty
    # if similar_chunks is None or len(similar_chunks) == 0:
    #     yield "No relevant information available to answer the question."

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
    """
        Generates a revised question based on the user's input question and the chat state.

        This function takes a user-provided question and the current chat state (a list of previous messages) 
        to formulate a more precise question that can be used to query a vector database. 

        Parameters:
        - question (str): The original question posed by the user.
        - chat_state (List[Dict]): A list of dictionaries representing the chat history, where each dictionary 
        contains a 'role' (either 'user' or 'assistant') and 'content' (the message text).
        - table_name (str): The name of the database table to be queried.

        Returns:
        - Generator[str, None, None]: A generator that yields strings, which are the responses from the 
        database query based on the revised question.

        The function constructs a prompt that instructs the model to generate a question suitable for querying 
        the vector database. It utilizes the last five messages from the chat state to provide context, 
        ensuring that only relevant prior messages are considered. The function logs the chat messages and 
        the revised question for debugging purposes.

        The revised question is then used to create an embedding, which is subsequently used to perform a 
        cosine similarity search against the specified database table. The results of this search are streamed 
        back to the user.

        Example:
        If the user asks, "What are the implications of data privacy laws?", the function may revise this 
        to a more specific question like "How do data privacy laws affect small businesses?" before querying 
        the database.
    """
    chat_messages = chat_state[-5:][::-1] + [{"role": "user", "content": question}, {"role": "assistant", "content": "I will follow the instructions."}]

    revision_prompt = f"""
        Question: {question}
        When possible, rewrite the question using <chat history> to identify the intent of the question, the people referenced by the question, and ideas / topics / themes targeted by the question in <chat history>.
        If the <chat history> does not contain any information about the people, ideas, or topics relevant to the question, then do not make any assumptions.
        Only return the request. Don't preface it or provide an introductory message.
    """

    # ---
    #     Example
    #     ---
    #     '''What are his best ideas?'''
    #     becomes
    #     '''What are <person's name>'s best idea about <topic mentioned in chat history>? Consider how this might affect areas mentioned in <a prior chat answer>.'''
    #     ---

    revision_system_instructions = "You are an assistant that concisely and carefully rewrites questions. The less than (<) and greater than (>) signs are telling you to refer to the chat history. Don't use < or > in your response."

    vectordb_prompt = f"""
        Request: {question}\n\nBased on this request, what request should I make to my vector database?
        Use prior messages to establish the intent and context of the question. 
        Include any relevant topics, themes, or individuals mentioned in the chat history. 
        Significantly lengthen the request and include as many contextual details as possible to enhance the relevance of the query.
        Only return the request. Don't preface it or provide an introductory message.
    """

    # Examples below within quotes
    #     ---
    #     Original Question: "What are the implications of data privacy laws?"
    #     New Question: "Considering the discussions about <specific data privacy topics or incidents> in prior messages, what are the implications of data privacy laws for <specific context or industry>? Include references to <related chat history details or previous discussions>."

    #     Original Question: "Summarize the latest podcast episode."
    #     New Question: "Provide a detailed summary of the main themes discussed in the latest podcast episode about <specific topic>. Include insights from <mentioned speaker> and relate it to <related discussion or theme in chat history>. Explain how <specific idea or concept> was elaborated in the episode."

    #     Original Question: "Make a list on best practices for managing remote teams."
    #     New Question: "Create a comprehensive list of best practices for managing remote teams. Consider the strategies discussed in <related prior messages>. This list should include methods that address <specific challenges or themes mentioned previously>."
    #     ---

    vectordb_system_instructions = "You expand on questions asked to a vector database containing chunks of transcripts. You add sub-questions and contextual details to make the query more specific and relevant to the chat history."

    fallback_model_order = [
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
        }
    ]  

    revised_question = fallback_text_response(
        prompt=revision_prompt,
        model_order=fallback_model_order,
        history_messages=chat_messages,
        system_instructions=revision_system_instructions,
    )

    logger.info(f"\n\nRevised question: {revised_question}\nModel: {fallback_model_order[0]['model']}\n\n")

    vectordb_question = fallback_text_response(
        prompt=vectordb_prompt,
        model_order=fallback_model_order,
        history_messages=chat_messages,
        system_instructions=vectordb_system_instructions,
    )
    logger.info(f"\n\nVector database question: {vectordb_question}\nModel: {fallback_model_order[0]['model']}\n\n")

    query_embedding = create_openai_embedding(
        text=vectordb_question, 
        model_choice="text-embedding-3-large"
    )
    similar_chunks = cosine_similarity_search(
        table_name=table_name, 
        query_embedding=query_embedding,
    )

    return single_question(question=revised_question, similar_chunks=similar_chunks)
    

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


