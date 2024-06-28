import pandas as pd
import numpy as np
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.types import TypeDecorator, UserDefinedType
from dotenv import load_dotenv
import os
import logging
from typing import Any, Dict, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vector(TypeDecorator):
    impl = UserDefinedType

    def get_col_spec(self):
        return "vector"

    def bind_expression(self, bindvalue):
        return bindvalue

    def column_expression(self, col):
        return col

def load_env_variables() -> Dict[str, str]:
    logger.info("Loading environment variables")
    load_dotenv()
    env_vars = {
        "sql_user": os.getenv("SQL_USER"),
        "sql_password": os.getenv("SQL_PASSWORD"),
        "sql_host": os.getenv("SQL_HOST"),
        "sql_database": os.getenv("SQL_DATABASE")
    }
    logger.debug(f"Loaded environment variables: {', '.join(env_vars.keys())}")
    return env_vars

def create_connector() -> Any:
    logger.info("Creating Google Cloud SQL connector")
    return Connector()

def get_connection(env_vars: Dict[str, str], connector: Any) -> Callable[[], Any]:
    logger.info("Creating database connection function")
    def getconn():
        logger.debug("Establishing database connection")
        conn = connector.connect(
            instance_connection_string=env_vars["sql_host"],
            user=env_vars["sql_user"],
            password=env_vars["sql_password"],
            db=env_vars["sql_database"],
            driver="pg8000"
        )
        logger.debug("Database connection established")
        return conn
    return getconn

def create_engine(getconn: Callable[[], Any]) -> Any:
    logger.info("Creating SQLAlchemy engine")
    engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)
    logger.debug("SQLAlchemy engine created")
    return engine

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

def ensure_table_schema(
    engine: Any, 
    table_name: str, 
    df: pd.DataFrame
) -> None:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        with engine.begin() as conn:
            columns = []
            for column, dtype in df.dtypes.items():
                if isinstance(df[column].iloc[0], (np.ndarray, list)):
                    sql_type = Vector()
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
                        sql_type = Vector()
                    else:
                        dtype = df[column].dtype
                        sql_type = 'VARCHAR(255)' if dtype == 'object' else 'FLOAT' if dtype == 'float64' else 'INT'
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"))
            logging.info(f"Added columns {new_columns} to {table_name}")

def write_to_table(engine: Any, table_name: str, data_object: Dict[str, Any]) -> None:
    logger.info(f"Writing data to table '{table_name}'")
    df = pd.DataFrame(data_object)
    
    try:
        with engine.begin() as connection:
            ensure_table_schema(engine, table_name, df)
            
            columns = ', '.join(df.columns)
            placeholders = ', '.join([f':{col}' for col in df.columns])
            insert_stmt = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            for index, row in df.iterrows():
                logger.debug(f"Inserting row {index + 1} of {len(df)}")
                row_dict = {
                    col: f"[{','.join(map(str, row[col]))}]" if isinstance(row[col], (np.ndarray, list)) else row[col]
                    for col in df.columns
                }
                
                stmt = text(insert_stmt)
                connection.execute(stmt, row_dict)
                
        logger.info(f"Successfully appended {len(df)} rows to table '{table_name}'")
    except Exception as e:
        logger.error(f"Error writing to table '{table_name}': {e}", exc_info=True)
        raise

def read_from_table(engine: Any, table_name: str) -> pd.DataFrame:
    logger.info(f"Reading data from table '{table_name}'")
    try:
        with engine.connect() as connection:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, connection)
        logger.info(f"Successfully read {len(df)} rows from table '{table_name}'")
        return df
    except Exception as e:
        logger.error(f"Error reading from table '{table_name}': {e}", exc_info=True)
        raise

def main():
    logger.info("Starting main function")
    env_vars = load_env_variables()
    connector = create_connector()
    try:
        getconn = get_connection(env_vars, connector)
        engine = create_engine(getconn)
        logger.info("Engine created successfully")
        
        ensure_pgvector_extension(engine)

        table_name = 'vector_table'
        data_object = {
            'id': [1, 2, 3],
            'text': ['example1', 'example2', 'example3'],
            'embedding': [np.random.rand(128).tolist() for _ in range(3)]
        }

        write_to_table(engine, table_name, data_object)
        df = read_from_table(engine, table_name)
        print(df)
        logger.info(f"Read {len(df)} rows from table '{table_name}'")
        logger.debug(f"DataFrame head:\n{df.head()}")
        
        #delete_table(engine, table_name)
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Closing connector")
        connector.close()
    logger.info("Main function completed")

if __name__ == "__main__":
    main()