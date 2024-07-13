CONFIG = {
    'feed_url': "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ",
    'start_date': "January 1, 2023",
    'end_date': "February 1, 2023",
    'process_new_episodes': True,
    'generate_embeddings': True,
    'write_to_table': True
}

PODCAST_CONFIG = {
    'feed_url': CONFIG['feed_url'],
    'start_date': CONFIG['start_date'],
    'end_date': CONFIG['end_date'],
    'audio_dir_name': "tmp_audio",
    'output_dir_name': "tmp",
    'output_file_name': "chunks_to_embed.json"
}

EMBEDDING_CONFIG = {
    'input_podcast_file': PODCAST_CONFIG['output_file_name'],
    'input_podcast_dir': PODCAST_CONFIG['output_dir_name'],
    'existing_embeddings_file': "embedded_chunks.json",
    'existing_embeddings_dir': "tmp",
    'output_file_name': "embedded_chunks.json",
    'output_dir_name': "tmp",
    'delete_input_file': False
}

TABLE_CONFIG = {
    'table_name': "vector_table",
    'input_file_name': EMBEDDING_CONFIG['output_file_name'],
    'input_dir_name': EMBEDDING_CONFIG['output_dir_name'],
    'delete_input_file': False
}