import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.SQL_USER = os.getenv("SQL_USER")
        self.SQL_PASSWORD = os.getenv("SQL_PASSWORD")
        self.SQL_HOST = os.getenv("SQL_HOST")
        self.SQL_DATABASE = os.getenv("SQL_DATABASE")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    def __str__(self):
        return f"Config(SQL_USER={self.SQL_USER}, SQL_HOST={self.SQL_HOST}, SQL_DATABASE={self.SQL_DATABASE}, LOG_LEVEL={self.LOG_LEVEL})"