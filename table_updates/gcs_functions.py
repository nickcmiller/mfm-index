import json
from google.cloud import storage
import logging
from typing import Dict, Any, Optional, List
from google.cloud.exceptions import NotFound
from genai_toolbox.helper_functions.datetime_helpers import get_date_with_timezone
from datetime import datetime
from dateutil import parser
from datetime import timezone

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
    if not bucket_name or not file_path:
        raise ValueError("bucket_name and file_path are required")

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

def create_empty_index(
    bucket_name: str
) -> None:
    """
    Create an empty index.
    """
    upload_json_to_gcs(
        json_data={},
        bucket_name=bucket_name,
        file_path="index.json"
    )

def process_and_upload_entries(
    entries: list, 
    bucket_name: str
):
    """
    Process entries, upload to GCS, and update the index.

    :param entries: List of entries to process
    :param bucket_name: Name of the GCS bucket
    """
    index_additions = []
    for entry in entries:
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

def retrieve_index_list_from_gcs(
    bucket_name: str, 
    file_path: str
) -> Optional[List[Dict[str, Any]]]:
    """
        Retrieve a JSON file from Google Cloud Storage and return its content as a list of dictionaries.

        Args:
            bucket_name (str): Name of the GCS bucket
            file_path (str): Path and filename of the object in GCS (e.g., 'folder/subfolder/file.json')

        Returns:
            Optional[List[Dict[str, Any]]]: The content of the file as a list of dictionaries, or None if retrieval fails
    """
    dict_content = retrieve_dict_from_gcs(bucket_name, file_path)
    if dict_content is None:
        return None
    return list(dict_content.values())

def retrieve_entries_by_id(
    bucket_name: str, 
    video_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve full entries using the index from Google Cloud Storage.

    Args:
        bucket_name (str): Name of the GCS bucket
        video_ids (Optional[List[str]]): List of video IDs to retrieve. If None, retrieve all entries.

    Returns:
        List[Dict[str, Any]]: List of retrieved full entries
    """
    # Retrieve the index as a list
    index_list = retrieve_index_list_from_gcs(bucket_name, "index.json")
    if index_list is None:
        logging.error("Failed to retrieve index")
        return []

    # Filter the index list based on video_ids if provided
    if video_ids is not None:
        index_list = [entry for entry in index_list if entry['video_id'] in video_ids]

    # Retrieve full entries
    full_entries = []
    for entry in index_list:
        file_path = f"{entry['channel_id']}/{entry['video_id']}.json"
        full_entry = retrieve_dict_from_gcs(bucket_name, file_path)
        if full_entry is not None:
            full_entries.append(full_entry)
        else:
            logging.warning(f"Failed to retrieve entry for video_id: {entry['video_id']}")

    return full_entries

def filter_index_by_date_range(
    index_list: List[Dict[str, Any]], 
    start_datetime: datetime, 
    end_datetime: datetime, 
    timezone_str: str,
    date_field: str = 'published'
) -> List[Dict[str, Any]]:
    """
    Filter and sort entries based on the specified date range.

    Args:
        index_list (List[Dict[str, Any]]): List of entries to filter
        start_datetime (datetime): Start of the date range
        end_datetime (datetime): End of the date range
        timezone_str (str): Timezone for the date range
        date_field (str): The field to use for the date (default is 'published')

    Returns:
        List[Dict[str, Any]]: Filtered and sorted list of entries within the date range
    """
    filtered_index = []
    for entry in index_list:
        published_date = get_date_with_timezone(entry[date_field], timezone_str)
        if start_datetime <= published_date <= end_datetime:
            filtered_index.append(entry)

    filtered_index.sort(key=lambda entry: get_date_with_timezone(entry[date_field], timezone_str))
    return filtered_index

def retrieve_entries_by_date_range(
    bucket_name: str,
    start_date: str,
    end_date: str,
    timezone_str: str = 'UTC',
    date_field: str = 'published'
) -> List[Dict[str, Any]]:
    """
    Retrieve full entries from Google Cloud Storage within a specified date range.

    Args:
        bucket_name (str): Name of the GCS bucket
        start_date (str): Start of the date range (inclusive) as a string
        end_date (str): End of the date range (inclusive) as a string
        timezone_str (str): Timezone for the date range (default is 'UTC')

    Returns:
        List[Dict[str, Any]]: List of retrieved full entries within the date range
    """
    # Convert start and end dates to timezone-aware datetime objects
    try:
        start_datetime = get_date_with_timezone(start_date, timezone_str)
        end_datetime = get_date_with_timezone(end_date, timezone_str)
    except Exception as e:
        logging.error(f"Error converting dates: {e}")
        return []

    # Retrieve the index as a list
    index_list = retrieve_index_list_from_gcs(bucket_name, "index.json")
    if index_list is None:
        logging.error("Failed to retrieve index")
        return []

    # Filter the index list based on the date range
    filtered_index = filter_index_by_date_range(index_list, start_datetime, end_datetime, timezone_str, date_field)

    # Retrieve full entries
    full_entries = []
    for entry in filtered_index:
        file_path = f"{entry['channel_id']}/{entry['video_id']}.json"
        full_entry = retrieve_dict_from_gcs(bucket_name, file_path)
        if full_entry is not None:
            full_entries.append(full_entry)
        else:
            logging.warning(f"Failed to retrieve entry for video_id: {entry['video_id']}")

    return full_entries

    
if __name__ == "__main__":
    from genai_toolbox.helper_functions.string_helpers import retrieve_file, write_to_file
    import json

    # index_list = retrieve_index_list_from_gcs(
    #     bucket_name="aai_utterances_json",
    #     file_path="index.json"
    # )
    # filtered_index = filter_index_by_date_range(
    #     index_list=index_list,
    #     start_datetime=datetime(2024, 3, 28, tzinfo=timezone.utc),
    #     end_datetime=datetime(2024, 8, 1, tzinfo=timezone.utc),
    #     timezone_str='UTC'
    # )
    # print(json.dumps(filtered_index, indent=4))
    entries = retrieve_entries_by_date_range(
        bucket_name="aai_utterances_json",
        start_date="2024-02-1",
        end_date="2024-06-1"
    )
    print(f"Retrieved {len(entries)} entries from GCS")
    output_file_path = "tmp/speaker_replaced_utterances.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(entries, output_file, indent=4)
    print(f"Successfully wrote {len(entries)} entries to {output_file_path}")
    # Convert entries to include parsed dates for sorting

    entries_with_dates = [
        (parser.isoparse(entry['published']), entry) for entry in entries
    ]
    print(entries[0].keys())
    for _, entry in entries_with_dates:
        published_date = parser.isoparse(entry['published'])  # Corrected method name
        day_with_suffix = f"{published_date.day}{'th' if 11 <= published_date.day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(published_date.day % 10, 'th')}"
        formatted_date = f"{published_date.strftime('%B')} {day_with_suffix}, {published_date.year}"
        print(f"{formatted_date} - {entry['title']}")


    


