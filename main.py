from config import CONFIG, PODCAST_CONFIG, EMBEDDING_CONFIG, QUERY_CONFIG
from podcast_processor import process_podcast_feed
from embedding_generator import generate_embeddings
from query_handler import handle_query

from cloud_sql_gcp.config.gcp_sql_config import load_config
from cloud_sql_gcp.databases.connection import get_db_engine
from cloud_sql_gcp.databases.operations import ensure_pgvector_extension, write_list_of_objects_to_table, read_from_table, delete_table
from cloud_sql_gcp.utils.logging import setup_logging

def main():
    if CONFIG['process_new_episodes']:
        process_podcast_feed(PODCAST_CONFIG)
    
    if CONFIG['generate_embeddings']:
       response = generate_embeddings(EMBEDDING_CONFIG)
       for utterance in response:
           print(utterance['id'])
    if CONFIG['run_query']:
        response = handle_query(QUERY_CONFIG)

        question = QUERY_CONFIG['question']

        print(f"Number of query responses: {len(response['query_response'])}")
        # print(json.dumps(response['query_response'], indent=4))
        print(f"\n\nQuestion: {question}\n\n")
        print(f"Response: {response['llm_response']}\n\n")

if __name__ == "__main__":
    main()
    