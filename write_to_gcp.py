import pandas as pd
import numpy as np
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import inspect, text
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_env_variables():
    load_dotenv()
    return {
        "sql_user": os.getenv("SQL_USER"),
        "sql_password": os.getenv("SQL_PASSWORD"),
        "sql_host": os.getenv("SQL_HOST"),
        "sql_database": os.getenv("SQL_DATABASE")
    }

# Register the numpy array type with psycopg2
def addapt_numpy_array(numpy_array):
    return AsIs(repr(numpy_array.tolist()))

def create_connector():
    return Connector()

def get_connection(env_vars, connector):
    def getconn():
        conn = connector.connect(
            instance_connection_string=env_vars["sql_host"],
            user=env_vars["sql_user"],
            password=env_vars["sql_password"],
            db=env_vars["sql_database"],
            driver="pg8000"  # Changed to pg8000 for PostgreSQL
        )
        return conn
    return getconn

def create_engine(getconn):
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",  # Changed to PostgreSQL
        creator=getconn,
    )

def ensure_pgvector_extension(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logging.info("pgvector extension enabled")
    except Exception as e:
        logging.error(f"Error ensuring pgvector extension: {e}")
        raise 

def ensure_table_schema(engine, table_name, df):
    """
        Ensures the table schema matches the DataFrame structure.

        This function checks if the specified table exists in the database. If it doesn't,
        it creates the table with columns matching the DataFrame's structure. If the table
        exists, it adds any new columns present in the DataFrame but not in the table.

        Args:
            engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the database.
            table_name (str): The name of the table to check or create.
            df (pandas.DataFrame): The DataFrame containing the data structure to match.

        Raises:
            Exception: If there's an error during table creation or column addition.

        Note:
            - For vector columns (numpy arrays or lists), it uses the pgvector 'vector' type.
            - For other data types, it uses VARCHAR(255) for objects, FLOAT for float64, and INT for integers.
    """
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        with engine.begin() as conn:
            columns = []
            for column, dtype in df.dtypes.items():
                if isinstance(df[column].iloc[0], (np.ndarray, list)):
                    vector_dim = len(df[column].iloc[0])
                    sql_type = f"vector({vector_dim})"
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
                        vector_dim = len(df[column].iloc[0])
                        sql_type = f"vector({vector_dim})"
                    else:
                        dtype = df[column].dtype
                        sql_type = 'VARCHAR(255)' if dtype == 'object' else 'FLOAT' if dtype == 'float64' else 'INT'
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"))
            logging.info(f"Added columns {new_columns} to {table_name}")

def write_to_table(engine, table_name, data_object):
    df = pd.DataFrame(data_object)
    
    try:
        with engine.begin() as connection:
            ensure_table_schema(engine, table_name, df)
            
            columns = ', '.join(df.columns)
            placeholders = ', '.join([f':{col}' for col in df.columns])
            insert_stmt = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            for _, row in df.iterrows():
                # Convert numpy array to properly formatted vector string
                row_dict = {
                    col: f"[{','.join(map(str, row[col]))}]" if isinstance(row[col], (np.ndarray, list)) else row[col]
                    for col in df.columns
                }
                
                stmt = text(insert_stmt)
                connection.execute(stmt, row_dict)
                
        logging.info(f"Data appended to table {table_name}")
    except Exception as e:
        logging.error(f"Error writing to table {table_name}: {e}")
        logging.error(f"Error details: {str(e)}")
        raise

def read_from_table(engine, table_name):
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM {table_name}"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def delete_table(engine, table_name):
    with engine.connect() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    logging.info(f"Table {table_name} has been deleted")

def main():
    env_vars = load_env_variables()
    connector = create_connector()
    try:
        getconn = get_connection(env_vars, connector)
        engine = create_engine(getconn)
        logging.info("Engine created successfully")
        
        ensure_pgvector_extension(engine)

        table_name = 'vector_table'
        data_object = {
            'id': [1, 2, 3],
            'text': ['example1', 'example2', 'example3'],
            'embedding': [np.random.rand(128).tolist() for _ in range(3)]  # Convert to list
        }

        write_to_table(engine, table_name, data_object)
        df = read_from_table(engine, table_name)
        print(df)
        
        #delete_table(engine, table_name)
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
        logging.error(f"Error details: {str(e)}")
    finally:
        connector.close()

if __name__ == "__main__":
    main()