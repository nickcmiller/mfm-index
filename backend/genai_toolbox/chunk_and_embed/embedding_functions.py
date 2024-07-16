from genai_toolbox.clients.openai_client import openai_client
from typing import Dict, Callable

# Similarity Metrics
def cosine_similarity(
    vec1: list[float], 
    vec2: list[float]
):
    """
        Calculates the cosine similarity between two vectors.

        This function computes the cosine similarity between two vectors, which is a measure of
        the cosine of the angle between them. It is often used to determine how similar two vectors are,
        regardless of their magnitude.

        Args:
        vec1 (list[float]): The first vector, represented as a list of floats.
        vec2 (list[float]): The second vector, represented as a list of floats.

        Returns:
        float: The cosine similarity between vec1 and vec2. The value ranges from -1 to 1,
            where 1 indicates identical vectors, 0 indicates orthogonal vectors,
            and -1 indicates opposite vectors.

        Raises:
        ValueError: If the input vectors have different lengths.

        Note:
        This function uses numpy for efficient vector operations.
    """
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Embedding Clients
def create_openai_embedding(
    model_choice: str,
    text: str, 
    client = openai_client()
) -> Dict:
    response = client.embeddings.create(
        input=text, 
        model=model_choice
    )
    return response.data[0].embedding

# Embedding Functions
def create_embedding_for_dict(
    embedding_function: Callable,
    chunk_dict: dict,
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-small"
) -> dict:
    """
        Creates an embedding for the text in the given dictionary using the specified model and retains all other key-value pairs.

        Args:
        chunk_dict (dict): A dictionary containing the text to embed under the key 'text' and possibly other data.
        embedding_function (Callable): The embedding function used to create embeddings.
        model_choice (str): The model identifier to use for embedding generation.

        Returns:
        dict: A dictionary containing the original text, its corresponding embedding, and all other key-value pairs from the input dictionary.
    """
    if 'text' not in chunk_dict:
        raise KeyError("The 'text' key is missing from the chunk_dict.")
    
    if not chunk_dict[key_to_embed]:
        raise ValueError(f"The '{key_to_embed}' value in chunk_dict is empty.")

    if not isinstance(embedding_function, Callable):
        raise ValueError("The 'embedding_function' argument must be a callable object.")

    text = chunk_dict[key_to_embed]
    embedding = embedding_function(
        model_choice=model_choice,
        text=text
    )
    result_dict = {**chunk_dict, "embedding": embedding}
    return result_dict

def embed_dict_list(
    embedding_function: Callable,
    chunk_dicts: list[dict],
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-small"
) -> list[dict]:
    """
        Creates embeddings for a list of dictionaries containing text.

        This function applies the specified embedding function to each dictionary in the input list,
        embedding the text found under the specified key in each dictionary.

        Args:
        embedding_function (Callable): The function used to create embeddings.
        chunk_dicts (list[dict]): A list of dictionaries, each containing text to be embedded.
        key_to_embed (str, optional): The key in each dictionary that contains the text to embed. Defaults to "text".
        model_choice (str, optional): The model identifier to use for embedding generation. Defaults to "text-embedding-3-small".

        Returns:
        list[dict]: A list of dictionaries, each containing the original data plus the generated embedding.

        Raises:
        ValueError: If the embedding_function is not callable, if the key_to_embed is missing from any dictionary,
                    or if the value for key_to_embed is empty in any dictionary.
    """
    return [create_embedding_for_dict(
        embedding_function=embedding_function, 
        chunk_dict=chunk_dict, 
        key_to_embed=key_to_embed,
        model_choice=model_choice
    ) for chunk_dict in chunk_dicts]

def add_similarity_to_next_dict_item(
    chunk_dicts: list[dict],
    similarity_metric: Callable = cosine_similarity
) -> list[dict]:
    """
        Adds a 'similarity_to_next_item' key to each dictionary in the list,
        calculating the cosine similarity between the current item's embedding
        and the next item's embedding. The last item's similarity is always 0.

        Args:
            chunk_dicts (list[dict]): List of dictionaries containing 'embedding' key.

        Returns:
            list[dict]: The input list with added 'similarity_to_next_item' key for each dict.

        Example:
            Input:
            [
                {..., "embedding": [0.1, 0.2, 0.3]},
                {..., "embedding": [0.4, 0.5, 0.6]},
            ]
            Output:
            [
                {..., "embedding": [0.1, 0.2, 0.3], "similarity_to_next_item": 0.9},
                {..., "embedding": [0.4, 0.5, 0.6], "similarity_to_next_item": 0.9},
            ]
    """
    for i in range(len(chunk_dicts) - 1):
        current_embedding = chunk_dicts[i]['embedding']
        next_embedding = chunk_dicts[i + 1]['embedding']
        similarity = cosine_similarity(current_embedding, next_embedding)
        chunk_dicts[i]['similarity_to_next_item'] = similarity

    # similarity_to_next_item for the last item is always 0
    chunk_dicts[-1]['similarity_to_next_item'] = 0

    return chunk_dicts

# Query embeddings
def find_similar_chunks(
    query: str,
    chunks_with_embeddings: list[dict], 
    embedding_function: Callable = create_openai_embedding,
    model_choice: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_returned_chunks: int = 10,
) -> list[dict]:
    query_embedding = embedding_function(text=query, model_choice=model_choice)

    similar_chunks = []
    for chunk in chunks_with_embeddings:
        similarity = cosine_similarity(query_embedding, chunk['embedding'])
        if similarity > threshold:
            chunk['similarity'] = similarity
            similar_chunks.append(chunk)
        
    similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    top_chunks = similar_chunks[0:max_returned_chunks]
    no_embedding_key_chunks = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in top_chunks]
    
    return no_embedding_key_chunks
