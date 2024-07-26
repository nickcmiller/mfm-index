import json
from google.cloud import storage
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)

def upload_json_to_gcs(
    json_data: dict, 
    bucket_name: str, 
    file_path: str,
    metadata: dict = None
) -> str:
    """
        Upload a JSON object to Google Cloud Storage.

        This function takes a dictionary, converts it to a JSON string, and uploads it
        to a specified location in Google Cloud Storage. It also allows for optional
        metadata to be attached to the uploaded file.

        Args:
            json_data (dict): The data to be uploaded as JSON.
            bucket_name (str): The name of the GCS bucket to upload to.
            file_path (str): The path and filename for the object in GCS.
            metadata (dict, optional): Metadata to attach to the uploaded file.

        Returns:
            str: The file path of the uploaded object in GCS.

        Raises:
            ValueError: If any of json_data, bucket_name, or file_path are not provided.
            GoogleCloudError: If there's an error during the upload process.

        Note:
            This function requires the google-cloud-storage library and
            appropriate GCP credentials to be set up.
    """
    if not all([json_data, bucket_name, file_path]):
        raise ValueError("json_data, bucket_name, and file_path are required")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    try:
        json_string = json.dumps(json_data, indent=2)
        blob.upload_from_string(json_string, content_type='application/json')

        if metadata:
            blob.metadata = metadata
            blob.patch()

        logging.info(f"File {file_path} uploaded to {bucket_name}{' with metadata' if metadata else ''}")
        return file_path
    except GoogleCloudError as e:
        logging.error(f"Error uploading to GCS: {e}")
        raisev

def retrieve_dict_from_gcs(
    bucket_name: str, 
    file_path: str
) -> Optional[Dict[str, Any]]:
    """
        Retrieve a JSON file from Google Cloud Storage and return its content as a dictionary.

        Args:
            bucket_name (str): Name of the GCS bucket
            file_path (str): Path and filename of the object in GCS (e.g., 'folder/subfolder/file.json')

        Returns:
            Optional[Dict[str, Any]]: The content of the file as a dictionary, or None if retrieval fails

        Raises:
            ValueError: If bucket_name or file_path is empty
    """
    if not bucket_name or not file_path:
        raise ValueError("Both bucket_name and file_path must be non-empty strings")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    try:
        file_content = blob.download_as_text()
        return json.loads(file_content)
    except NotFound:
        logging.error(f"File {file_path} does not exist in bucket {bucket_name}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from {file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error retrieving {file_path} from {bucket_name}: {e}")
    
    return None

def process_and_upload_utterances(utterances_data: list, bucket_name: str):
    """
    Process utterances data, upload to GCS, and update the index.

    :param utterances_data: List of utterance entries to process
    :param bucket_name: Name of the GCS bucket
    """
    index_additions = []
    for entry in utterances_data:
        metadata = {
            "video_id": entry['video_id'],
            "channel_id": entry['channel_id'],
            "title": entry['title'],
            "publisher": entry['feed_title'],
            "published": entry['published'],
        }
        logging.info(f"Uploading to GCS:\n\n{json.dumps(metadata, indent=4)}")
        file_path = f"{entry['channel_id']}/{entry['video_id']}.json"
        upload_json_to_gcs(
            json_data=entry,
            metadata=metadata,
            bucket_name=bucket_name,
            file_path=file_path
        )
        logging.info(f"Added {metadata['video_id']} to index")
        index_additions.append(metadata)

    index = retrieve_dict_from_gcs(
        bucket_name=bucket_name,
        file_path="index.json"
    )
    logging.info(f"Retrieved {len(index)} entries from {bucket_name}")
    for metadata in index_additions:
        index[metadata['video_id']] = metadata
    
    upload_json_to_gcs(
        json_data=index,
        bucket_name=bucket_name,
        file_path="index.json"
    )
    logging.info(f"Updated index with {len(index_additions)} new entries")

    
if __name__ == "__main__":
    from genai_toolbox.helper_functions.string_helpers import retrieve_file
    import json

    utterances_data = retrieve_file(
        file="speaker_replaced_utterances.json",
        dir_name="tmp"
    )
    print(f"Retrieved {len(utterances_data)} entries to process")

    bucket_name = "aai_utterances_json"
    process_and_upload_utterances(utterances_data, bucket_name)