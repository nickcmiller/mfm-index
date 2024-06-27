import mysql.connector
import pandas as pd
import pymysql
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import inspect, text
from dotenv import load_dotenv
import os
import logging

def load_env_variables():
    load_dotenv()
    return {
        "sql_user": os.getenv("SQL_USER"),
        "sql_password": os.getenv("SQL_PASSWORD"),
        "sql_host": os.getenv("SQL_HOST"),
        "sql_database": os.getenv("SQL_DATABASE")
    }

def create_connector():
    return Connector()

def get_connection(env_vars, connector):
    def getconn():
        conn = connector.connect(
            instance_connection_string=env_vars["sql_host"],
            user=env_vars["sql_user"],
            password=env_vars["sql_password"],
            db=env_vars["sql_database"],
            driver="pymysql"
        )
        return conn
    return getconn

def create_engine(getconn):
    return sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )

def ensure_table_schema(engine, table_name, df):
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
    else:
        existing_columns = set()
    new_columns = set(df.columns) - existing_columns

    if new_columns:
        with engine.begin() as conn:
            for column in new_columns:
                dtype = df[column].dtype
                sql_type = 'VARCHAR(255)' if dtype == 'object' else 'FLOAT' if dtype == 'float64' else 'INT'
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"))
        logging.info(f"Added columns {new_columns} to {table_name}")

def write_to_table(engine, table_name, data_object):
    df = pd.DataFrame(data_object)
    
    try:
        ensure_table_schema(engine, table_name, df)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logging.info(f"Data appended to table {table_name}")
    except Exception as e:
        logging.error(f"Error writing to table {table_name}: {e}")
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
    getconn = get_connection(env_vars, connector)
    engine = create_engine(getconn)

    table_name = 'new_table'
    data_object = {'column1': [1, 2, 3], 'column2': ['e', 'f', 'g'], 'column3': ['h', 'i', 'j']}

    write_to_table(engine, table_name, data_object)
    df = read_from_table(engine, table_name)
    print(df)
    
    #delete_table(engine, table_name)
    connector.close()

if __name__ == "__main__":
    main()