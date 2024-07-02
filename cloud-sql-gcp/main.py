import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from genai_toolbox.helper_functions.string_helpers import retrieve_file

import numpy as np
import os
import json
import logging
from typing import Any, Dict, Callable, List, Optional

from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy import inspect, text, select
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import insert

from databases.connection import create_connector, get_connection, create_engine
from config.gcp_sql_config import Config, load_config
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from databases.operations import ensure_pgvector_extension, ensure_table_schema, write_to_table, write_list_of_objects_to_table, read_from_table, delete_table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



@contextmanager
def get_db_engine(config: Config):
    connector = create_connector()
    try:
        getconn = get_connection(config, connector)
        engine = create_engine(getconn)
        yield engine
    finally:
        connector.close()

@contextmanager
def get_db_connection(engine):
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def main():
    logger.info("Starting main function")
    config = load_config()
    
    try:
        with get_db_engine(config) as engine:
            logger.info("Engine with connection pooling created successfully")
            
            ensure_pgvector_extension(engine)


            # try:
            #     write_to_table(engine, table_name, data_object)
            # except Exception as e:
            #     logger.error(f"Failed to write to table: {e}", exc_info=True)
            #     return

            try:
                write_list_of_objects_to_table(engine, table_name, list_of_objects, batch_size=1000)
            except Exception as e:
                logger.error(f"Failed to write to table: {e}", exc_info=True)
                return

            try:
                rows = read_from_table(engine, table_name)
                logger.info(f"Read {len(rows)} rows from table '{table_name}'")
                if rows:
                    print("Available keys in the first row:", rows[0].keys())
                    print("\nRow contents:")
                    for index, row in enumerate(rows, start=1):
                        print(f"\nRow {index}:")
                        for key, value in row.items():
                            print(f"  {key}: {type(value)}")
                            if key == 'embedding':
                                print(f"    Length: {len(value)}")
                            elif isinstance(value, (str, int, float)):
                                print(f"    Value: {value}")
                            elif isinstance(value, list) and len(value) > 0:
                                print(f"    First element: {value[0]}")
                            else:
                                print(f"    Type: {type(value)}")
            except Exception as e:
                logger.error(f"Failed to read from table: {e}", exc_info=True)
            
            
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    logger.info("Main function completed")

if __name__ == "__main__":
    import json 
    table_name = 'vector_table'
    aggregated_chunked_embeddings = retrieve_file(
        file="aggregated_chunked_embeddings.json",
        dir_name="../tmp"
    )
    print(aggregated_chunked_embeddings[0].keys()) #dict_keys(['speakers', 'text', 'embedding', 'title', 'start_time', 'end_time'])
    
    data_object = aggregated_chunked_embeddings[0]
    filtered_data_object = {key: value for key, value in data_object.items() if key != 'embedding'}
    for key, value in filtered_data_object.items():
        print(f"{key}: {type(value)}")
    print(json.dumps(filtered_data_object, indent=4))
    list_of_objects = aggregated_chunked_embeddings[:3]
    if True:
        main()
    else:
        logger.info("Deleting table")
        try:
            config = load_config()
            with get_db_engine(config) as engine:
                delete_table(engine, table_name)
            logger.info(f"Table '{table_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting table '{table_name}': {e}", exc_info=True)
