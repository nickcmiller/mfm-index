from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from typing import List, Dict, Any

from genai_toolbox.helper_functions.string_helpers import retrieve_file, write_to_file
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding

from gcp_postgres_pgvector.config.gcp_sql_config import load_config
from gcp_postgres_pgvector.databases.connection import get_db_engine
from gcp_postgres_pgvector.databases.operations import ensure_pgvector_extension, write_list_of_objects_to_table, read_from_table, read_similar_rows, delete_table, delete_rows_by_ids
from gcp_postgres_pgvector.utils.logging import setup_logging

logger = setup_logging()
config = load_config()

def initialize_database(
    table_name, 
    _,
    config=config, 
):
    """
        Initialize the database by creating a connection and ensuring the pgvector extension is installed.

        This function creates a database engine with connection pooling and ensures that the pgvector
        extension is available in the database. The pgvector extension is required for efficient
        storage and querying of vector embeddings.

        Args:
            config (dict): A dictionary containing the database configuration parameters.
            table_name (str): The name of the table (not used in this function, but included for consistency).
            _ (Any): Placeholder for unused parameter (maintained for function signature consistency).

        Raises:
            Exception: If there's an error during the database initialization process.
    """
    with get_db_engine(config) as engine:
        logger.info("Engine with connection pooling created successfully")
        ensure_pgvector_extension(engine)
    logger.info("Database initialized successfully")

def write_to_table(
    table_name, 
    list_of_objects,
    config=config
) -> None:
    with get_db_engine(config) as engine:
        try:
            ensure_pgvector_extension(engine)
            write_list_of_objects_to_table(
                engine, 
                table_name, 
                list_of_objects, 
                batch_size=1000,
                unique_column='id'
            )
            logger.info(f"Successfully wrote to table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to write to table: {e}", exc_info=True)
            raise

def read_from_table_and_log(
    table_name,
    _, 
    config=config,
    keep_embeddings=False
) -> Dict[str, Any]:
    """
        Read data from the specified table and log the results.

        This function connects to the database, reads all rows from the specified table,
        logs information about the data read, and returns the rows based on the keep_embeddings flag.

        Args:
            config (dict): A dictionary containing the database configuration parameters.
            table_name (str): The name of the table to read from.
            _ (Any): Placeholder for unused parameter (maintained for function signature consistency).
            keep_embeddings (bool): Flag to determine whether to keep embeddings in the returned data.

        Returns:
            dict: A dictionary containing one or two keys:
                - 'rows': A list of all rows read from the table, including embeddings if keep_embeddings is True.
                - 'non_embedding_rows': A list of all rows with the 'embedding' field removed (only if keep_embeddings is False).

        Raises:
            Exception: If there's an error during the database read operation.
    """
    with get_db_engine(config) as engine:
        ensure_pgvector_extension(engine)
        try:
            rows = read_from_table(engine, table_name, include_embedding=keep_embeddings)

            if not rows: 
                logger.info(f"No data found in table '{table_name}'")
                return {'rows': [], 'non_embedding_rows': []}

            logger.info(f"Read {len(rows)} rows from table '{table_name}'")
            logger.info(f"Available keys in the first row: {rows[0].keys()}")

            result = {'rows': rows}

            # Deduplicate rows based on 'title' and keep the latest 'date_published'
            unique_rows = {}
            for row in rows:
                title = row['title']
                if title not in unique_rows or row['date_published'] > unique_rows[title]['date_published']:
                    unique_rows[title] = {}
                    unique_rows[title]['id'] = row['id']
                    unique_rows[title]['title'] = row['title']
                    unique_rows[title]['date_published'] = row['date_published']
            
            sorted_unique_rows = sorted(unique_rows.values(), key=lambda x: x['date_published'])
            for row in sorted_unique_rows:
                logger.info(f"{row['date_published']} - {row['title']}")

            if not keep_embeddings:
                non_embedding_rows = [{k: v for k, v in row.items() if k != 'embedding'} for row in rows]
                logger.info(f"Row non-embedding values: {json.dumps(non_embedding_rows[:1], indent=4)}")
                result['non_embedding_rows'] = non_embedding_rows
            else:
                for row in rows:
                    logger.info(f"Embedding type: {type(row['embedding'])}")
                    logger.info(f"Embedding sample: {row['embedding'][:1]}")

            return result
        except Exception as e:
            logger.error(f"Failed to read from table: {e}", exc_info=True)
            raise

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
            ensure_pgvector_extension(engine)
            similar_rows = read_similar_rows(engine, table_name, query_embedding, limit=limit)
            logger.info(f"Found {len(similar_rows)} similar rows in table '{table_name}'")
            
            # for row in similar_rows:
            #     logger.info(f"Similarity: {row['similarity']}, ID: {row['id']}")
            #     logger.info(f"Text: {row['text'][:100]}...")
            
            return similar_rows
        except Exception as e:
            logger.error(f"Failed to perform cosine similarity search: {e}", exc_info=True)
            raise

def delete_table_if_exists(
    table_name,
    _,
    config=config
) -> None:
    """
        Delete the specified table if it exists in the database.

        This function attempts to delete the given table from the database. If the table
        doesn't exist, it will log an info message and continue without raising an error.

        Args:
            config (dict): A dictionary containing the database configuration parameters.
            table_name (str): The name of the table to be deleted.
            _ (Any): Placeholder for unused parameter (maintained for function signature consistency).

        Raises:
            Exception: If there's an error during the table deletion process, other than
                    the table not existing.
    """
    logger.info(f"Deleting table '{table_name}'")
    with get_db_engine(config) as engine:
        try:
            ensure_pgvector_extension(engine)
            delete_table(engine, table_name)
            logger.info(f"Table '{table_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting table '{table_name}': {e}", exc_info=True)

def delete_rows_with_ids(
    table_name: str,
    ids: List[Any],
    config=config
) -> int:
    """
        Delete rows from the specified table based on a list of IDs.

        Args:
            table_name (str): The name of the table to delete rows from.
            ids (List[Any]): A list of IDs to delete.
            config (dict): A dictionary containing the database configuration parameters.

        Returns:
            int: The number of rows deleted.

        Raises:
            Exception: If there's an error during the deletion process.
    """
    print(f"Config in delete_rows_by_ids: {config}")
    logger.info(f"Deleting rows from table '{table_name}' with IDs: {ids}")
    with get_db_engine(config) as engine:
        try:
            ensure_pgvector_extension(engine)
            deleted_count = delete_rows_by_ids(engine, table_name, ids)
            logger.info(f"Successfully deleted {deleted_count} rows from table '{table_name}'")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete rows from table '{table_name}': {e}", exc_info=True)
            raise

def load_and_process_data(
    file_name: str = "test_embeddings.json",
    dir_name: str = "tmp"
) -> List[Dict[str, Any]]:
    """
        Load and process data from a JSON file.

        This function retrieves data from a JSON file named 'test_embeddings.json' in the 'tmp' directory,
        processes it, and returns a subset of the data for further use. The function prints the types of each key-value pair in the first data object (excluding 'embedding'). The full data is loaded, but only a subset is returned to limit memory usage and processing time.

        Returns:
            list: A list containing the first 6 items from the loaded and processed data.
    """
    aggregated_chunked_embeddings = retrieve_file(
        file=file_name,
        dir_name=dir_name
    )
    
    data_object = aggregated_chunked_embeddings[0]
    filtered_data_object = {key: value for key, value in data_object.items() if key != 'embedding'}
    for key, value in filtered_data_object.items():
        logger.info(f"{key}: {type(value)}")
    return aggregated_chunked_embeddings

def save_similar_search_to_json(
    table_name,
    query_text,
    model_choice="text-embedding-3-large",
    config=config,
    limit=100,
    dir_name="tmp",
    file_name="similar_search_results.json"
):
    """
        Save the results of a cosine similarity search to a JSON file.

        This function performs a cosine similarity search on the specified table and saves the results to a JSON file.
        The results are stored in a dictionary with the IDs as keys and the texts as values.

        Args:
            table_name (str): The name of the table to search in.
            query_text (str): The text to compare against.
            model_choice (str): The model to use for embedding.
            dir_name (str): The directory to save the results to.
            file_name (str): The name of the file to save the results to.
            config (dict): A dictionary containing the database configuration parameters.
            limit (int): The maximum number of similar rows to return.
    """

    query_embedding = create_openai_embedding(
        text=query_text,
        model_choice=model_choice
    )

    search_results = cosine_similarity_search(
        table_name, 
        query_embedding,
        config=config,
        limit=limit
    )

    search_dict = {}
    for result in search_results:
        search_dict[result['id']] = result['text']

    write_to_file(
        search_dict,
        file_name=file_name,
        dir_name=dir_name
    )

def delete_json_results(
    dir_name="tmp",
    file_name="similar_search_results.json"
):
    results = retrieve_file(
        file_name=file_name,
        dir_name=dir_name
    )
    ids = list(results.keys())
    
    delete_rows_with_ids(
        table_name, 
        ids, 
        config=config
    )


def main(
    operation, 
    query_embedding=None,
    table_name=None
) -> None:
    logger.info(f"Starting main function with operation: {operation}")
    config = load_config()
    
    operations = {
        'init': initialize_database,
        'read': read_from_table_and_log,
        'write': write_to_table,
        'delete': delete_table_if_exists,
        'similarity': cosine_similarity_search
    }
    
    if operation not in operations:
        logger.error(f"Invalid operation: {operation}")
        return
    
    try:        
        if operation == 'similarity':
            if query_embedding is None:
                # Generate a random query embedding if none is provided
                query_embedding = np.random.rand(3072).tolist()  # Assuming 3072-dimensional embeddings
                logger.info("Using a random query embedding for similarity search")
            result = operations[operation](table_name, query_embedding, config=config)
        elif operation == 'read':
            result = operations[operation](table_name, None, config=config)
            result = result['non_embedding_rows'][:1]
        else:
            result = operations[operation](table_name, None, config=config)
        
        if result:
            logger.info(f"Operation result: {json.dumps(result, indent=4)}")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    
    logger.info("Main function completed")

if __name__ == "__main__":
    table_name = 'vector_table'

    # import argparse

    # parser = argparse.ArgumentParser(description="Process some arguments.")
    # parser.add_argument("operation", type=str, help="Specify the operation to perform")

    # args = parser.parse_args()

    # main(args.operation, table_name='vector_table')

    if False:
        text = "Let's get back to the show"

        save_similar_search_to_json(
            table_name,
            query_text=text,
            file_name="similar_search_results.json",
            dir_name="tmp"
        )
    else:
        delete_json_results(
            file_name="similar_search_results.json",
            dir_name="tmp"
        )