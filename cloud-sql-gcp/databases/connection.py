import logging
import sqlalchemy
from sqlalchemy.pool import QueuePool
from typing import Any, Callable
from google.cloud.sql.connector import Connector

from config.gcp_sql_config import Config

logger = logging.getLogger(__name__)

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