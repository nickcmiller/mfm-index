from trufflepig import Trufflepig, Index
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional

load_dotenv()

trufflepig_api_key = os.getenv("TRUFFLEPIG_API_KEY")
client = Trufflepig(trufflepig_api_key)

def initialize_index(index_name: str):
    index = client.get_index(index_name)
    if not index:
        index = client.create_index(index_name)
    return index

def list_indexes() -> Dict[str, Index]:
    return client.list_indexes()

def delete_index(index_name: str):
    return client.delete_index(index_name)

def create_document(file_path:str, metadata: Dict[str, str] = {}) -> Dict[str, str]:

    file_name = os.path.basename(file_path)
    metadata.update({"filename": file_name})

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    return {
        "document_path": file_path,
        "metadata": metadata
    }

def upload_documents(index: str, files: List[str]):
    return index.upload(files=files)

def upload_string(
    index: str, 
    document_key: str, 
    string: str, 
    metadata: Dict[str, str] = {}
) -> Dict[str, Optional[str]]:
    return index.upload_string(document_key, string, metadata)

def get_upload_status(index: str, files: List[str]):
    job_tracking_response = index.get_upload_status(files)
    return job_tracking_response

def list_documents(index: str):
    return index.list_documents()

def delete_document(index: str, document_key: str):
    return index.delete_document(document_key)

def search_index(index: str, query: str, max_results: int = 3):
    return index.search(query_text=query, max_results=max_results)

if __name__ == "__main__":
    import json
    
    index_name = "mfm-index"
    index = initialize_index(index_name)

    # created_document = create_document("replaced_transcript.txt", {"authors": ["Sam Parr", "Shaan Puri"]})
    # print(created_document)

    # response = upload_documents(index, [created_document])
    # print(response)

    print(list_documents(index))
    
    for result in search_index(index, "What life advice does he have?"):
        print(json.dumps(result, indent=4))
        print(f"{result.content}\n")

    print()