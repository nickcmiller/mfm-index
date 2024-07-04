from config import CONFIG, PODCAST_CONFIG, EMBEDDING_CONFIG, QUERY_CONFIG
from podcast_processor import process_podcast_feed
from embedding_generator import generate_embeddings
from query_handler import handle_query

def main():
    if CONFIG['process_new_episodes']:
        process_podcast_feed(CONFIG, PODCAST_CONFIG)
    
    if CONFIG['generate_embeddings']:
        generate_embeddings(EMBEDDING_CONFIG)
    if CONFIG['run_query']:
        response = handle_query(QUERY_CONFIG)

        question = QUERY_CONFIG['question']

        print(f"Number of query responses: {len(response['query_response'])}")
        # print(json.dumps(response['query_response'], indent=4))
        print(f"\n\nQuestion: {question}\n\n")
        print(f"Response: {response['llm_response']}\n\n")

if __name__ == "__main__":
    main()