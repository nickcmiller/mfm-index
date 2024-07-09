CONFIG = {
    'feed_url': "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ",
    'start_date': "November 1, 2023",
    'end_date': "December 1, 2023",
    'process_new_episodes': True,
    'generate_embeddings': True,
    'write_to_table': True,
    'run_query': False,
}

PODCAST_CONFIG = {
    'feed_url': CONFIG['feed_url'],
    'start_date': CONFIG['start_date'],
    'end_date': CONFIG['end_date'],
    'audio_dir_name': "tmp_audio",
    'output_dir_name': "tmp",
    'output_file_name': "new_chunks.json",
}

EMBEDDING_CONFIG = {
    'input_podcast_file': PODCAST_CONFIG['output_file_name'],
    'input_podcast_dir': PODCAST_CONFIG['output_dir_name'],
    'existing_embeddings_file': "test_embeddings.json",
    'existing_embeddings_dir': "tmp",
    'output_file_name': "test_embeddings.json",
    'output_dir_name': "tmp"
}

TABLE_CONFIG = {
    'table_name': "vector_table",
    'input_file_name': EMBEDDING_CONFIG['output_file_name'],
    'input_dir_name': EMBEDDING_CONFIG['output_dir_name']
}

QUERY_CONFIG = {
    'question': "What do Ben and John think of the Vision Pro?",
    'input_file_name': EMBEDDING_CONFIG['output_file_name'],
    'input_dir_name': EMBEDDING_CONFIG['output_dir_name']
}