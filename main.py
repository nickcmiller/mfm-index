from config import CONFIG, PODCAST_CONFIG, EMBEDDING_CONFIG, TABLE_CONFIG, QUERY_CONFIG
from podcast_processor import process_podcast_feed
from embedding_generator import generate_embeddings
from sql_operations import write_to_table
from query_handler import handle_query

from genai_toolbox.helper_functions.string_helpers import retrieve_file

def main():
    if CONFIG['process_new_episodes']:
        process_podcast_feed(PODCAST_CONFIG)
    
    if CONFIG['generate_embeddings']:
       response = generate_embeddings(EMBEDDING_CONFIG)
       for utterance in response:
           print(utterance['id'])
    
    if CONFIG['write_to_table']:
        list_of_objects = retrieve_file(
            file=TABLE_CONFIG['input_file_name'], 
            dir_name=TABLE_CONFIG['input_dir_name']
        )

        write_to_table(
            table_name=TABLE_CONFIG['table_name'],
            list_of_objects=list_of_objects
        )

    # if CONFIG['run_query']:
    #     response = handle_query(QUERY_CONFIG)

    #     question = QUERY_CONFIG['question']

    #     print(f"Number of query responses: {len(response['query_response'])}")
    #     # print(json.dumps(response['query_response'], indent=4))
    #     print(f"\n\nQuestion: {question}\n\n")
    #     print(f"Response: {response['llm_response']}\n\n")

if __name__ == "__main__":
    main()
    