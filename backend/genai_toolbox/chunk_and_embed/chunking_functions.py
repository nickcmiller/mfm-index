from typing import Callable, List, Dict
import re

# Helper Functions
def add_metadata_to_chunks(
    chunks: list[dict],
    additional_metadata: dict = {}
) -> list[dict]:

    return [
        {    
            **{k: v for k, v in chunk.items()},
            **additional_metadata
        } for chunk in chunks
    ]

# Splitting Text Document Text into Chunks
def split_text_string(
    text: str, 
    separator: str
) -> list[dict]:
    """
        Splits the text by the given separator and returns a list of dictionaries where each dictionary has a key 'text' with non-empty chunks as values.

        Args:
        text (str): The text to split.
        separator (str): The separator to use for splitting the text.

        Returns:
        list[dict]: A list of dictionaries with the key 'text' and values as non-empty text chunks.
    """
    chunks = text.split(separator)
    return [{"text": chunk} for chunk in chunks if chunk]

def consolidate_split_chunks(
    chunk_dicts: list[dict], 
    min_length: int = 75
) -> list[dict]:
    """
        Combines consecutive chunks of text that are shorter than `max_length`.
        Chunks longer than `max_length` are added as separate entries in the result list.

        Args:
        chunk_dicts (list of dict): The list of dictionaries containing text chunks to process.
        max_length (int): The maximum length for a chunk to be considered short.

        Returns:
        list of dict: A new list of dictionaries where short consecutive chunks have been combined.
    """
    lengthened_chunk_dicts = []
    combine_chunks = []

    for chunk_dict in chunk_dicts:
        chunk = chunk_dict['text']
        if len(chunk) == 0:
            continue

        if len(chunk) < min_length:
            combine_chunks.append(chunk)
        else:
            if combine_chunks:
                combine_chunks.append(chunk)
                lengthened_chunk_dicts.append({"text": '\n\n'.join(combine_chunks)})
                combine_chunks = []
            else:
                lengthened_chunk_dicts.append({"text": chunk})

    if combine_chunks:
        lengthened_chunk_dicts.append({"text": '\n\n'.join(combine_chunks)})

    return lengthened_chunk_dicts

def consolidate_similar_split_chunks(
    chunks: list[dict], 
    threshold: float = 0.45
) -> list[dict]:
    """
        Consolidates similar chunks based on their precomputed 'similarity_to_next_item'.

        Args:
            chunks (list[dict]): List of dictionaries, each containing 'text' and 'similarity_to_next_item' keys.
            threshold (float): Similarity threshold for consolidation.

        Returns:
            list[dict]: Consolidated list of dictionaries, each with a 'text' key.
    """
    consolidated_chunks = []
    current_chunk_texts = []

    for i, chunk in enumerate(chunks):
        current_chunk_texts.append(chunk['text'])

        if chunk['similarity_to_next_item'] < threshold or i == len(chunks) - 1:
            consolidated_chunks.append({"text": '\n\n'.join(current_chunk_texts)})
            current_chunk_texts = []

    return consolidated_chunks

# AssemblyAI Utterance Chunking
def convert_utterance_speaker_to_speakers(
    utterances: list[dict]
) -> list[dict]:

    mod_utterances = []
    for utterance in utterances:
        speakers = [utterance['speaker']]
        mod_utterances.append({
            **{k: v for k, v in utterance.items() if k != 'speaker'},
            "speakers": speakers
        })
        
    return mod_utterances

def consolidate_short_utterances(
    utterances: list[dict],
    min_length: int = 75
) -> list[dict]:
    """
        Consolidates short utterances from AssemblyAI transcription into longer segments.

            Args:
                utterances (list[dict]): A list of dictionaries, each representing an utterance
                    with keys 'confidence', 'end', 'speaker', 'start', and 'text'.
            min_length (int, optional): The minimum length of text to be considered a
                standalone utterance. Defaults to 75 characters.

        Returns:
            list[dict]: A list of consolidated utterances.
    """
    consolidated = []
    current_group = None

    def finalize_group():
        if current_group:
            if len(set(current_group["speakers"])) > 1:
                text = "\n\n".join(f"{{}}: {t}" for t in current_group["texts"])
            else:
                text = current_group["texts"][0]
            
            consolidated.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "speakers": current_group["speakers"],
                "text": text,
            })

    for utterance in utterances:
        utterance_text = utterance['text'].strip()
        if not utterance_text:
            continue

        if current_group is None:
            current_group = {
                "start": utterance['start'],
                "end": utterance['end'],
                "speakers": [utterance['speaker']],
                "texts": [utterance_text],
            }
        else:
            current_group["end"] = utterance['end']
            current_group["speakers"].append(utterance['speaker'])
            current_group["texts"].append(utterance_text)

        if len(utterance_text) >= min_length:
            finalize_group()
            current_group = None

    finalize_group()  # Handle the last group if exists

    return consolidated

def consolidate_similar_utterances(
    utterances: list[dict],
    similarity_threshold: float = 0.45
) -> list[dict]:
    """
        Consolidates similar utterances based on their similarity to the next item.

        Args:
            utterances (list[dict]): List of utterances, each containing 'text', 'speakers', 
                                    'similarity_to_next_item', and other fields.
            similarity_threshold (float): Threshold for considering utterances similar.

        Returns:
            list[dict]: Consolidated list of utterances.
    """
    consolidated = []
    current_group = None

    def finalize_group():
        if current_group:
            consolidated.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "speakers": current_group["speakers"],
                "text": "\n\n".join(current_group["texts"]),
            })

    def format_text(text, speakers):
        return f"{{}}: {text}" if len(speakers) == 1 else text

    for utterance in utterances:
        utterance_text = utterance['text'].strip()
        if not utterance_text:
            continue

        formatted_text = format_text(utterance_text, utterance['speakers'])

        if current_group is None:
            current_group = {
                "start": utterance['start'],
                "end": utterance['end'],
                "speakers": utterance['speakers'].copy(),
                "texts": [formatted_text],
            }
        else:
            current_group["end"] = utterance['end']
            current_group["speakers"].extend(utterance['speakers'])
            current_group["texts"].append(formatted_text)

        if utterance['similarity_to_next_item'] < similarity_threshold:
            finalize_group()
            current_group = None

    finalize_group()  # Handle the last group if exists

    return consolidated

def format_speakers_in_utterances(
    utterances: list[dict]
) -> list[dict]:
    """
        Formats the speakers in the given utterances.

        This function takes a list of utterance dictionaries and formats the speakers
        within the text of each utterance. If the text contains placeholders ('{}'),
        it replaces them with the corresponding speaker names. If no placeholders are
        present, the text is left unchanged.

        If speakers appear more than once in utterance['speakers'], the duplicates are removed.

        Args:
            utterances (list[dict]): A list of dictionaries, each representing an utterance.
                Each dictionary should contain 'speakers' and 'text' keys.

        Returns:
            list[dict]: A new list of dictionaries with the same structure as the input,
            but with the 'text' field formatted to include speaker names where applicable.

        Example:
            Input:
            [
                {'speakers': ['John'], 'text': '{}: Hello!'},
                {'speakers': ['Alice', 'Bob'], 'text': 'Nice to meet you.'}
            ]
            
            Output:
            [
                {'speakers': ['John'], 'text': 'John: Hello!'},
                {'speakers': ['Alice', 'Bob'], 'text': 'Nice to meet you.'}
            ]
    """
    def format_text(
        text: str, 
        speakers: list[str]
    ) -> str:
        if '{}' in text:
            return text.format(*speakers)
        return text

    return [
        {
            **{k: v for k, v in utterance.items() if k != 'speakers'},
            "text": format_text(utterance['text'], utterance['speakers']),
            "speakers": list(dict.fromkeys(utterance['speakers']))  # Remove duplicates
        }
        for utterance in utterances
    ]

def milliseconds_to_minutes_in_utterances(
    utterances: list[dict]
) -> list[dict]:
    """
    Adds minutes and seconds format timestamps to utterances while keeping millisecond values.

    This function takes a list of utterance dictionaries, each containing 'start' and 'end'
    keys with millisecond values, and adds new 'start_time' and 'end_time' keys with
    'minutes:seconds' format. The original 'start' and 'end' keys are retained.

    Args:
        utterances (list[dict]): A list of dictionaries, each representing an utterance.
            Each dictionary should contain 'start' and 'end' keys with millisecond values.

    Returns:
        list[dict]: A new list of dictionaries with the same structure as the input,
        but with additional 'start_time' and 'end_time' keys in 'minutes:seconds' format.

    Example:
        Input:
        [
            {'speakers': ['John'], 'text': 'Hello!', 'start': 1000, 'end': 2000},
            {'speakers': ['Alice'], 'text': 'Hi there!', 'start': 2500, 'end': 3500}
        ]
        
        Output:
        [
            {'speakers': ['John'], 'text': 'Hello!', 'start': 1000, 'end': 2000, 'start_time': '0:01', 'end_time': '0:02'},
            {'speakers': ['Alice'], 'text': 'Hi there!', 'start': 2500, 'end': 3500, 'start_time': '0:02', 'end_time': '0:03'}
        ]
    """
    def ms_to_min_sec(ms: int) -> str:
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"

    return [
        {
            **utterance,
            "start_mins": ms_to_min_sec(utterance['start']),
            "end_mins": ms_to_min_sec(utterance['end'])
        }
        for utterance in utterances
    ]

def rename_start_end_to_ms(
    utterances: list[dict]
) -> list[dict]:
    """
    Renames 'start' to 'start_ms' and 'end' to 'end_ms' in each utterance dictionary.

    Args:
        utterances (list[dict]): A list of dictionaries, each representing an utterance.
            Each dictionary should contain 'start' and 'end' keys.

    Returns:
        list[dict]: A new list of dictionaries with 'start' renamed to 'start_ms' and
        'end' renamed to 'end_ms'.

    Example:
        Input:
        [
            {'speakers': ['John'], 'text': 'Hello!', 'start': 1000, 'end': 2000},
            {'speakers': ['Alice'], 'text': 'Hi there!', 'start': 2500, 'end': 3500}
        ]
        
        Output:
        [
            {'speakers': ['John'], 'text': 'Hello!', 'start_ms': 1000, 'end_ms': 2000},
            {'speakers': ['Alice'], 'text': 'Hi there!', 'start_ms': 2500, 'end_ms': 3500}
        ]
    """
    return [
        {
            **{k: v for k, v in utterance.items() if k not in ['start', 'end']},
            'start_ms': utterance['start'],
            'end_ms': utterance['end']
        }
        for utterance in utterances
    ]

