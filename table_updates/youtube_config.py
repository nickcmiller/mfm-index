CONFIG = {
    'channel_id': "UCyaN6mg5u8Cjy2ZI4ikWaug",
    'start_date': "2024-07-10",
    'end_date': "2024-07-26",
    'process_new_episodes': False,
    'generate_embeddings': True,
    'write_to_table': True
}

YOUTUBE_CONFIG = {
    'channel_id': CONFIG['channel_id'],
    'start_date': CONFIG['start_date'],
    'end_date': CONFIG['end_date'],
    'audio_dir_name': "tmp_audio",
    'output_dir_name': "tmp",
    'output_file_name': "chunks_to_embed.json"
}

EMBEDDING_CONFIG = {
    'input_file': YOUTUBE_CONFIG['output_file_name'],
    'input_dir': YOUTUBE_CONFIG['output_dir_name'],
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