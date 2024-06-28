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
import logging
from typing import Any, Dict, Callable
from gcp_sql_config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type



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
def ensure_table_schema(engine: Any, table_name: str, df: pd.DataFrame) -> None:
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            with engine.begin() as conn:
                columns = []
                for column, dtype in df.dtypes.items():
                    if column == 'embedding':
                        sql_type = Vector(128)  # Adjust the dimension as needed
                    elif isinstance(df[column].iloc[0], (np.ndarray, list)):
                        sql_type = f"float[]"
                    else:
                        sql_type = 'VARCHAR(255)' if dtype == 'object' else 'FLOAT' if dtype == 'float64' else 'INT'
                    columns.append(f"{column} {sql_type}")
                columns_str = ', '.join(columns)
                conn.execute(text(f"CREATE TABLE {table_name} ({columns_str})"))
            logging.info(f"Created table {table_name}")
        else:
            existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
            new_columns = set(df.columns) - existing_columns

            if new_columns:
                with engine.begin() as conn:
                    for column in new_columns:
                        if isinstance(df[column].iloc[0], (np.ndarray, list)):
                            sql_type = f"float[]"
                        else:
                            dtype = df[column].dtype
                            sql_type = 'VARCHAR(255)' if dtype == 'object' else 'FLOAT' if dtype == 'float64' else 'INT'
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"))
                logging.info(f"Added columns {new_columns} to {table_name}")
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
)
def write_to_table(
    engine: Any,
    table_name: str,
    data_object: Dict[str, Any],
    batch_size: int = 1000
) -> None:
    logger.info(f"Writing data to table '{table_name}'")
    df = pd.DataFrame(data_object)
    try:
        with get_db_connection(engine) as connection:
            ensure_table_schema(engine, table_name, df)
            # Convert embedding column to list of floats if it's a string representation
            if 'embedding' in df.columns and isinstance(df['embedding'].iloc[0], str):
                df['embedding'] = df['embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            
            # Prepare the insert statement
            insert_stmt = insert(sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=engine))
            
            # Perform batch inserts
            total_rows = len(df)
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size].to_dict(orient='records')
                connection.execute(insert_stmt, batch)
                connection.commit()
                logger.debug(f"Inserted batch {i//batch_size + 1} ({min(i+batch_size, total_rows)}/{total_rows} rows)")
            
            logger.info(f"Successfully appended {total_rows} rows to table '{table_name}'")
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
    table_name: str
) -> pd.DataFrame:
    logger.info(f"Reading data from table '{table_name}'")
    try:
        with get_db_connection(engine) as connection:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, connection)
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

            table_name = 'vector_table'
            data_object = {
                'id': [1, 2, 3],
                'text': ['example1', 'example2', 'example3'],
                'embedding': [np.random.rand(128).tolist() for _ in range(3)]
            }

            try:
                write_to_table(engine, table_name, data_object, batch_size=500)
            except Exception as e:
                logger.error(f"Failed to write to table: {e}", exc_info=True)
                return

            try:
                df = read_from_table(engine, table_name)
                logger.info(f"Read {len(df)} rows from table '{table_name}'")
                logger.debug(f"DataFrame head:\n{df.head()}")
            except Exception as e:
                logger.error(f"Failed to read from table: {e}", exc_info=True)
            
            # delete_table(engine, table_name)
            
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    logger.info("Main function completed")

if __name__ == "__main__":
    main()