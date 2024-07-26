import json
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)

def upload_json_to_gcs(
    json_data: dict, 
    bucket_name: str, 
    file_path: str = None,
    metadata: dict = None
) -> str:
    """
    Upload JSON data and optional metadata to Google Cloud Storage.

    :param json_data: Dictionary containing the JSON data to upload
    :param bucket_name: Name of the GCS bucket
    :param file_path: Optional path and filename for the object in GCS (e.g., 'folder/subfolder/file.json')
    :param metadata: Optional dictionary containing metadata for the file
    :return: The file path of the uploaded object
    """
    # Create a client using default credentials
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob (object) in the bucket
    blob = bucket.blob(file_path)

    # Convert JSON data to string
    json_string = json.dumps(json_data, indent=2)

    # Upload the file
    blob.upload_from_string(json_string, content_type='application/json')

    # Set metadata if provided
    if metadata:
        blob.metadata = metadata
        blob.patch()

    print(f"File {file_path} uploaded to {bucket_name}" + (" with metadata." if metadata else "."))
    
    return file_path


    
if __name__ == "__main__":
    from genai_toolbox.helper_functions.string_helpers import retrieve_file
    import json

    utterances_data = retrieve_file(
        file="speaker_replaced_utterances.json",
        dir_name="tmp"
    )
    print(f"Retrieved {len(utterances_data)} entries to process")

    index_additions = []
    bucket_name = "aai_utterances_json"
    for entry in utterances_data:
        metadata = {
            "video_id": entry['video_id'],
            "channel_id": entry['channel_id'],
            "title": entry['title'],
            "publisher": entry['feed_title'],
            "published": entry['published'],
        }
        print(json.dumps(metadata, indent=4))
        file_path = f"{entry['channel_id']}/{entry['video_id']}.json"
        upload_json_to_gcs(
            json_data=entry,
            metadata=metadata,
            bucket_name=bucket_name,
            file_path=file_path
        )
        index_additions.append(metadata)

    index = {}
    for metadata in index_additions:
        index[metadata['video_id']] = metadata
        index_file_path = "index.json"
        new_index = index
        new_index[metadata['video_id']] = metadata
    upload_json_to_gcs(
        json_data=new_index,
        bucket_name=bucket_name,
        file_path=index_file_path
    )