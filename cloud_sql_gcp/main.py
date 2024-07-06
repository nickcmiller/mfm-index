import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

from genai_toolbox.helper_functions.string_helpers import retrieve_file
from config.gcp_sql_config import load_config
from databases.connection import get_db_engine
from databases.operations import ensure_pgvector_extension, write_list_of_objects_to_table, read_from_table, delete_table
from utils.logging import setup_logging

logger = setup_logging()

def main(operation):
    logger.info(f"Starting main function with operation: {operation}")
    config = load_config()
    
    operations = {
        'init': initialize_database,
        'read': read_from_table_and_log,
        'write': write_to_table,
        'delete': delete_table_if_exists
    }
    
    if operation not in operations:
        logger.error(f"Invalid operation: {operation}")
        return
    
    try:
        operations[operation](config, table_name, list_of_objects)
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    
    logger.info("Main function completed")

def initialize_database(config, table_name, _):
    with get_db_engine(config) as engine:
        logger.info("Engine with connection pooling created successfully")
        ensure_pgvector_extension(engine)
    logger.info("Database initialized successfully")

def write_to_table(config, table_name, list_of_objects):
    with get_db_engine(config) as engine:
        try:
            write_list_of_objects_to_table(
                engine, 
                table_name, 
                list_of_objects, 
                batch_size=1000
            )
            logger.info(f"Successfully wrote to table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to write to table: {e}", exc_info=True)
            raise

def read_from_table_and_log(config, table_name, _):
    with get_db_engine(config) as engine:
        try:
            rows = read_from_table(engine, table_name)
            logger.info(f"Read {len(rows)} rows from table '{table_name}'")
            logger.info(f"Available keys in the first row: {rows[0].keys()}")

            for row in rows:
                non_embedding_values = {k: v for k, v in row.items() if k != 'embedding'}
                logger.info(f"Row non-embedding values: {non_embedding_values}")
        except Exception as e:
            logger.error(f"Failed to read from table: {e}", exc_info=True)
            raise

def delete_table_if_exists(config, table_name, _):
    logger.info(f"Deleting table '{table_name}'")
    with get_db_engine(config) as engine:
        try:
            delete_table(engine, table_name)
            logger.info(f"Table '{table_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting table '{table_name}': {e}", exc_info=True)

def load_and_process_data():
    aggregated_chunked_embeddings = retrieve_file(
        file="test_embeddings.json",
        dir_name="../tmp"
    )
    print(aggregated_chunked_embeddings[0].keys())
    
    data_object = aggregated_chunked_embeddings[0]
    filtered_data_object = {key: value for key, value in data_object.items() if key != 'embedding'}
    for key, value in filtered_data_object.items():
        print(f"{key}: {type(value)}")
    print(json.dumps(filtered_data_object, indent=4))
    return aggregated_chunked_embeddings[:3]

if __name__ == "__main__":
    import json 
    import sys
    
    table_name = 'vector_table'
    list_of_objects = load_and_process_data()
    
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    else:
        operation = 'init'  # default operation
    
    main(operation)
