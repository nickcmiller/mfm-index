import pandas as pd
import numpy as np
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.types import TypeDecorator, UserDefinedType
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
from contextlib import contextmanager
from dotenv import load_dotenv
import os
import json
import logging
from typing import Any, Dict, Callable, List, Optional
from gcp_sql_config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from genai_toolbox.helper_functions.string_helpers import retrieve_file


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> Config:
    logger.info("Loading configuration")
    config = Config()
    logger.debug(f"Loaded configuration: {config}")
    return config

def create_connector() -> Any:
    logger.info("Creating Google Cloud SQL connector")
    return Connector()

def get_connection(
    config: Config, 
    connector: Any
) -> Callable[[], Any]:
    logger.info("Creating database connection function")
    def getconn():
        logger.debug("Establishing database connection")
        conn = connector.connect(
            instance_connection_string=config.SQL_HOST,
            user=config.SQL_USER,
            password=config.SQL_PASSWORD,
            db=config.SQL_DATABASE,
            driver="pg8000"
        )
        logger.debug("Database connection established")
        return conn
    return getconn

def create_engine(
    getconn: Callable[[], Any]
) -> Any:
    logger.info("Creating SQLAlchemy engine with connection pooling")
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    logger.debug("SQLAlchemy engine with connection pooling created")
    return engine

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def ensure_pgvector_extension(engine: Any) -> None:
    logger.info("Ensuring pgvector extension is enabled")
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled successfully")
    except Exception as e:
        logger.error(f"Error ensuring pgvector extension: {e}", exc_info=True)
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def ensure_table_schema(
    engine: Any, 
    table_name: str, 
    data_object: Dict[str, Any],
    vector_dimensions: int = 3072
) -> None:
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            with engine.begin() as conn:
                columns = []
                for column, value in data_object.items():
                    if column == 'embedding':
                        sql_type = Vector(vector_dimensions)
                    elif isinstance(value, str):
                        sql_type = "TEXT"
                    elif isinstance(value, (int, np.integer)):
                        sql_type = "INTEGER"
                    elif isinstance(value, (float, np.float64)):
                        sql_type = "FLOAT"
                    else:
                        sql_type = "TEXT"
                    columns.append(f"{column} {sql_type}")
                columns_str = ', '.join(columns)
                conn.execute(text(f"CREATE TABLE {table_name} ({columns_str})"))
            logging.info(f"Created table {table_name}")
        else:
            logging.info(f"Table {table_name} already exists")
    except Exception as e:
        logger.error(f"Error ensuring table schema: {e}", exc_info=True)
        raise

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




def serialize_complex_types(obj):
    if isinstance(obj, (list, dict, set)):
        return json.dumps(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def deserialize_complex_types(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    return obj

def write_to_table(
    engine: Any,
    table_name: str,
    data_object: Dict[str, Any],
    unique_columns: Optional[List[str]] = None
) -> None:
    logger.info(f"Writing data to table '{table_name}'")

    try:
        # Serialize complex types, except for 'embedding'
        serialized_data = {
            k: serialize_complex_types(v) if k != 'embedding' else v 
            for k, v in data_object.items()
        }

        # Convert embedding to list if it's a numpy array
        if 'embedding' in serialized_data and isinstance(serialized_data['embedding'], np.ndarray):
            serialized_data['embedding'] = serialized_data['embedding'].tolist()

        with get_db_connection(engine) as connection:
            # Ensure the table exists with the correct schema
            ensure_table_schema(engine, table_name, serialized_data)
            
            # Prepare the insert statement
            table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=engine)
            
            # Create the insert statement
            insert_stmt = insert(table).values(serialized_data)
            
            # If there are unique columns, create an upsert statement
            if unique_columns:
                update_dict = {c.name: c for c in insert_stmt.excluded if c.name not in unique_columns}
                insert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=unique_columns,
                    set_=update_dict
                )
                operation = 'upserted'
            else:
                operation = 'inserted'
            
            # Execute the insert/upsert
            connection.execute(insert_stmt)
            connection.commit()
            
            logger.info(f"Successfully {operation} 1 row to table '{table_name}'")
    except Exception as e:
        logger.error(f"Error writing to table '{table_name}': {e}", exc_info=True)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def write_list_of_objects_to_table(
    engine: Any,
    table_name: str,
    data_list: List[Dict[str, Any]],
    unique_columns: Optional[List[str]] = None,
    batch_size: int = 1000
) -> None:
    if not data_list:
        logger.info(f"No data to write to table '{table_name}'")
        return

    logger.info(f"Writing {len(data_list)} objects to table '{table_name}'")

    try:
        with engine.begin() as connection:
            ensure_table_schema(engine, table_name, data_list[0])
            
            table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=engine)
            insert_stmt = insert(table)
            
            if unique_columns:
                update_dict = {c.name: c for c in insert_stmt.excluded if c.name not in unique_columns}
                insert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=unique_columns,
                    set_=update_dict
                )
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                serialized_batch = []
                for data_object in batch:
                    # Serialize complex types, except for 'embedding'
                    serialized_data = {
                        k: serialize_complex_types(v) if k != 'embedding' else v 
                        for k, v in data_object.items()
                    }
                    
                    # Convert embedding to list if it's a numpy array
                    if 'embedding' in serialized_data and isinstance(serialized_data['embedding'], np.ndarray):
                        serialized_data['embedding'] = serialized_data['embedding'].tolist()
                    
                    serialized_batch.append(serialized_data)
                
                connection.execute(insert_stmt, serialized_batch)
                logger.info(f"Successfully processed {len(serialized_batch)} rows for table '{table_name}'")
        
        logger.info(f"Finished writing all {len(data_list)} objects to table '{table_name}'")
    except Exception as e:
        logger.error(f"Error writing to table '{table_name}': {e}", exc_info=True)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def read_from_table(
    engine: Any, 
    table_name: str,
    limit: int = None,
    where_clause: str = None
) -> pd.DataFrame:
    logger.info(f"Reading data from table '{table_name}'")
    try:
        with get_db_connection(engine) as connection:
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql(query, connection)
        
        # Deserialize complex types for all columns except 'embedding'
        for column in df.columns:
            if column != 'embedding':
                df[column] = df[column].apply(deserialize_complex_types)
        
        logger.info(f"Successfully read {len(df)} rows from table '{table_name}'")
        return df
    except Exception as e:
        logger.error(f"Error reading from table '{table_name}': {e}", exc_info=True)
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def delete_table(
    engine: Any, 
    table_name: str
) -> None:
    logger.info(f"Deleting table '{table_name}'")
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        logger.info(f"Successfully deleted table '{table_name}'")
    except Exception as e:
        logger.error(f"Error deleting table '{table_name}': {e}", exc_info=True)
        raise

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
                df = read_from_table(engine, table_name)
                logger.info(f"Read {len(df)} rows from table '{table_name}'")
                print(df.head())
                print(df['speakers'])
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
        dir_name="tmp"
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
