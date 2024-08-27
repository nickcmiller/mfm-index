from dotenv import load_dotenv
load_dotenv()

from youtube_config import CONFIG, YOUTUBE_CONFIG, SPEAKER_REPLACEMENT_CONFIG, EMBEDDING_CONFIG, TABLE_CONFIG
from youtube_processor import process_youtube_feed
from speaker_replacement import replace_speakers
from embedding_generator import generate_embeddings
from sql_operations import write_to_table
from genai_toolbox.helper_functions.string_helpers import retrieve_file, delete_file
import time
import logging

logging.basicConfig(level=logging.INFO)

def main():
    start_time = time.time()
    if CONFIG['process_new_episodes']:
        logging.info(f"\n{'_'*50}\nPROCESSING NEW EPISODES\n{'_'*50}")
        process_youtube_feed(YOUTUBE_CONFIG)
        logging.info(f"Finished processing new episodes in {time.time() - start_time} seconds")
        start_time = time.time()

    if CONFIG['speaker_replacement']:
        logging.info(f"\n{'_'*50}\nREPLACING SPEAKERS\n{'_'*50}")
        replace_speakers(SPEAKER_REPLACEMENT_CONFIG)
        logging.info(f"Finished replacing speakers in {time.time() - start_time} seconds")
        start_time = time.time()
    
    if CONFIG['generate_embeddings']:
        logging.info(f"\n{'_'*50}\nGENERATING EMBEDDINGS\n{'_'*50}")
        response = generate_embeddings(EMBEDDING_CONFIG)
        

        if EMBEDDING_CONFIG['delete_input_file']:
            delete_file(
                file=EMBEDDING_CONFIG['input_podcast_file'],
                dir_name=EMBEDDING_CONFIG['input_podcast_dir']
            )
        
        logging.info(f"Finished generating embeddings in {time.time() - start_time} seconds")
        start_time = time.time() 
    
    if CONFIG['write_to_table']:
        logging.info(f"\n{'_'*50}\nWRITING TO TABLE\n{'_'*50}")
        list_of_objects = retrieve_file(
            file_name=TABLE_CONFIG['input_file_name'], 
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

        logging.info(f"Finished writing to table in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()