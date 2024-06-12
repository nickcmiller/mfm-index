from trufflepig import Trufflepig, Index
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

trufflepig_api_key = os.getenv("TRUFFLEPIGG_API_KEY")
client = Trufflepig(trufflepig_api_key)

def initialize_index(index_name: str):
    index = client.get_index(index_name)
    if not index:
        index = client.create_index(index_name)
    return index

def delete_index(index_name: str):
    return client.delete_index(index_name)

def upload_documents(index: str, files: List[str]):
    document_keys = index.upload(files)
    return document_keys

if __name__ == "__main__":

    index_name = "test"
    index = initialize_index(index_name)

    print(index)
    print(type(index))