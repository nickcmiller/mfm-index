import mysql.connector
import pandas as pd
import pymysql
from google.cloud.sql.connector import Connector
import sqlalchemy

from dotenv import load_dotenv
import os

load_dotenv()

sql_user = os.getenv("SQL_USER")
sql_password = os.getenv("SQL_PASSWORD")
sql_host = os.getenv("SQL_HOST")
sql_database = os.getenv("SQL_DATABASE")

# Initialize the connector
connector = Connector()

# Function to create the database connection
print(f"type(sql_host): {type(sql_host)}")
def getconn():
    conn = connector.connect(
        instance_connection_string=str(sql_host),
        user=sql_user,
        password=sql_password,
        db=sql_database,
        driver="pymysql"
    )
    return conn

# Create the SQLAlchemy engine
engine = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

table_name = 'new_table'
data_object = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}
df_to_write = pd.DataFrame(data_object)

# Write the data to the table
df_to_write.to_sql(
    table_name, 
    engine, 
    if_exists='replace', 
    index=False
)
print(f"Data written to table {table_name}")

# Now read and print the data to verify
df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
print("Data in the table:")
print(df)

# Don't forget to close the connection when you're done
connector.close()