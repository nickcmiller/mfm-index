from config import CONFIG
from podcast_processor import process_podcast_feed
from embedding_generator import generate_embeddings
from query_handler import handle_query

def main():
    if CONFIG['process_new_episodes']:
        process_podcast_feed()
    
    if CONFIG['generate_embeddings']:
        generate_embeddings()
    
    if CONFIG['run_query']:
        handle_query(CONFIG['question'])

if __name__ == "__main__":
    main()