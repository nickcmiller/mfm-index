from typing import Any, Dict, List, Optional
import numpy as np
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy import inspect, text, select
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import insert

from ..utils.logging import setup_logging
from ..utils.serialization import serialize_complex_types, deserialize_complex_types
from ..utils.retry import db_retry_decorator

logger = setup_logging()

@db_retry_decorator()
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

@db_retry_decorator()
def ensure_table_schema(
    engine: Any, 
    table_name: str, 
    data_object: Dict[str, Any],
    unique_column: str = 'id',
    vector_dimensions: int = 3072
) -> None:
    try:
        if data_object is None:
            raise ValueError("data_object cannot be None")
        
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            with engine.begin() as conn:
                columns = []
                for column, value in data_object.items():
                    if column == 'embedding':
                        dim = len(value) if isinstance(value, list) else vector_dimensions
                        sql_type = Vector(dim)
                    elif isinstance(value, str):
                        sql_type = "TEXT"
                    elif isinstance(value, (int, np.integer)):
                        sql_type = "INTEGER"
                    elif isinstance(value, (float, np.float64)):
                        sql_type = "FLOAT"
                    else:
                        sql_type = "TEXT"
                    columns.append(f"{column} {sql_type}")
                
                columns.append(f"UNIQUE ({unique_column})")
                
                columns_str = ', '.join(columns)
                conn.execute(text(f"CREATE TABLE {table_name} ({columns_str})"))
            logger.info(f"Created table {table_name}")
        else:
            constraints = inspector.get_unique_constraints(table_name)
            unique_constraint_exists = any(unique_column in constraint['column_names'] for constraint in constraints)
            
            if not unique_constraint_exists:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD CONSTRAINT {table_name}_{unique_column}_key UNIQUE ({unique_column})"))
                logger.info(f"Added unique constraint on '{unique_column}' column for table '{table_name}'")
            else:
                logger.info(f"Unique constraint on '{unique_column}' already exists for table '{table_name}'")
            
            logger.info(f"Table {table_name} already exists with necessary constraints")
    except Exception as e:
        logger.error(f"Error ensuring table schema: {e}", exc_info=True)
        raise

@db_retry_decorator()
def write_list_of_objects_to_table(
    engine: Any,
    table_name: str,
    data_list: List[Dict[str, Any]],
    unique_column: str = 'id',
    batch_size: int = 1000
) -> None:
    if not data_list:
        logger.info(f"No data to write to table '{table_name}'")
        return

    logger.info(f"Writing {len(data_list)} objects to table '{table_name}'")

    try:
        with engine.begin() as connection:
            if data_list:
                ensure_table_schema(engine, table_name, data_list[0], unique_column)
            
            table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=engine)
            insert_stmt = insert(table)
            
            update_dict = {c.name: c for c in insert_stmt.excluded if c.name != unique_column}
            insert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=[unique_column],
                set_=update_dict
            )
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                serialized_batch = []
                for data_object in batch:
                    serialized_data = {
                        k: serialize_complex_types(v) if k != 'embedding' else v 
                        for k, v in data_object.items()
                    }
                    
                    if 'embedding' in serialized_data and isinstance(serialized_data['embedding'], np.ndarray):
                        serialized_data['embedding'] = serialized_data['embedding'].tolist()
                    
                    serialized_batch.append(serialized_data)
                
                connection.execute(insert_stmt, serialized_batch)
                logger.info(f"Successfully processed {len(serialized_batch)} rows for table '{table_name}'")
        
        logger.info(f"Finished writing all {len(data_list)} objects to table '{table_name}'")
    except Exception as e:
        logger.error(f"Error writing to table '{table_name}': {e}", exc_info=True)
        raise


@db_retry_decorator()
def read_from_table(
    engine: Any, 
    table_name: str,
    limit: int = None,
    where_clause: str = None
) -> List[Dict[str, Any]]:
    logger.info(f"Reading data from table '{table_name}'")
    try:
        with engine.connect() as connection:
            query = select(text("*")).select_from(text(table_name))
            if where_clause:
                query = query.where(text(where_clause))
            if limit:
                query = query.limit(limit)
            
            result = connection.execute(query)
            rows = [row._asdict() for row in result]

        # Deserialize complex types for all columns except 'embedding'
        for row in rows:
            for key, value in row.items():
                if key != 'embedding':
                    row[key] = deserialize_complex_types(value)

        logger.info(f"Successfully read {len(rows)} rows from table '{table_name}'")
        return rows
    except Exception as e:
        logger.error(f"Error reading from table '{table_name}': {e}", exc_info=True)
        raise

@db_retry_decorator()
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