from config import CONFIG, PODCAST_CONFIG, EMBEDDING_CONFIG, TABLE_CONFIG
from podcast_processor import process_podcast_feed
from embedding_generator import generate_embeddings
from sql_operations import write_to_table

from genai_toolbox.helper_functions.string_helpers import retrieve_file, delete_file

def main():
    if CONFIG['process_new_episodes']:
        process_podcast_feed(PODCAST_CONFIG)
    
    if CONFIG['generate_embeddings']:
        response = generate_embeddings(EMBEDDING_CONFIG)

        if EMBEDDING_CONFIG['delete_input_file']:
            delete_file(
                file=EMBEDDING_CONFIG['input_podcast_file'],
                dir_name=EMBEDDING_CONFIG['input_podcast_dir']
            )   
    
    if CONFIG['write_to_table']:
        list_of_objects = retrieve_file(
            file=TABLE_CONFIG['input_file_name'], 
            dir_name=TABLE_CONFIG['input_dir_name']
        )

        write_to_table(
            table_name=TABLE_CONFIG['table_name'],
            list_of_objects=list_of_objects
        )

        if TABLE_CONFIG['delete_input_file']:
            delete_file(
                file=TABLE_CONFIG['input_file_name'],
                dir_name=TABLE_CONFIG['input_dir_name']
            )

if __name__ == "__main__":
    main()
    