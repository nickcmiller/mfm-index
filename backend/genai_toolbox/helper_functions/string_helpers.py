import re
import os
import json
from pathlib import Path
import logging
import traceback

from typing import List, Optional, Union
import mimetypes

logging.basicConfig(level=logging.INFO)

def retrieve_file(
    file: str,
    dir_name: str = None
) -> Optional[Union[str, dict, list]]:
    """
    Retrieves the content from a file saved using write_to_file.

    Args:
        file (str): The name or path of the file.
        dir_name (str, optional): The directory name where the file is located.

    Returns:
        Optional[Union[str, dict, list]]: The content of the file, or None if there was an error.

    Raises:
        IOError: If there is an error opening or reading the file.
    """
    try:
        if dir_name:
            full_path = os.path.join(dir_name, file)
        elif os.path.dirname(file):
            full_path = file
        else:
            full_path = os.path.join(os.getcwd(), file) 

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
    except IOError as e:
        logging.error(f"retrieve_file failed to read file {full_path}: {e}")
        logging.error(traceback.format_exc())
        return None

def write_to_file(
    content: str | dict | list, 
    file: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
    output_dir_name: str = None
) -> Path:
    """
        Writes content to a file.

        Args:
            content (str | dict | list): The content to write to the file. Can be a string, a dictionary, or a list.
            file (str): The name of the file to write to.
            mode (str, optional): The mode to open the file in. Defaults to 'w'.
            encoding (str, optional): The encoding to use when writing the file. Defaults to 'utf-8'.
            output_dir_name (str, optional): The name of the output directory. If provided, the file will be created in this directory.

        Returns:
            Path: The path to the written file.

        Raises:
            ValueError: If the file argument is not a string or if the content is not a string, dictionary, or list.
            IOError: If there is an error writing to the file.
    """
    if not isinstance(file, str):
        raise ValueError("file must be a string.")
    if not isinstance(content, (str, dict, list)):
        raise ValueError("content must be a string, a dictionary, or a list.")

    # Determine the directory and create it if it doesn't exist
    if output_dir_name:
        directory = Path(os.getcwd()) / output_dir_name
    else:
        directory = Path(os.path.dirname(file) or os.getcwd())
    directory.mkdir(parents=True, exist_ok=True)

    # Create the full file path
    if output_dir_name:
        safe_title = ''.join(char if char.isalnum() or char in " -_" else '_' for char in Path(file).stem)
        extension = Path(file).suffix or '.txt'
        file_path = directory / f"{safe_title}{extension}"
    else:
        file_path = directory / Path(file).name

    try:
        with open(file_path, mode, encoding=encoding) as f:
            if isinstance(content, dict) or isinstance(content, list):
                json.dump(content, f, indent=4)
            else:
                f.write(content)
        logging.info(f"Successfully written to {file_path}")
        return file_path
    except IOError as e:
        logging.error(f"Failed to write to file {file_path}: {e}")
        raise

def delete_file(
    file: str,
    dir_name: str = None
) -> None:
    if dir_name:
        full_path = os.path.join(dir_name, file)
    elif os.path.dirname(file):
        full_path = file
    else:
        full_path = os.path.join(os.getcwd(), file) 

    if os.path.exists(full_path):
        os.remove(full_path) 

# Sorting files with numbers
def extract_leading_number_from_filename(
    filename: str
) -> int:
    """
        Extracts the leading number from a filename.

        Args:
            filename (str): The name of the file.
            
        Returns:
            int: The leading number extracted from the filename, or 0 if no leading number is found.

        Raises:
            ValueError: If the filename does not start with a number.
    """
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename '{filename}' does not start with a number.")

def sort_folder_of_txt_files(
    folder_path: str,
    ascending: bool = True
) -> List[str]:
    """
        Retrieve a list of sorted full file paths for all .txt files in the given folder path.

        Args:
            folder_path (str): The path to the folder containing the files.
            ascending (bool): Sort order of the files, True for ascending, False for descending.

        Returns:
            List[str]: A list of full file paths for all .txt files in the folder, sorted in specified order.

        Raises:
            ValueError: If the provided folder path does not exist or is not a directory.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory")

    try:
        file_pattern = re.compile(r'^\d+')
        text_files = [
            f for f in os.listdir(folder_path)
            if f.endswith('.txt') and file_pattern.match(f)
        ]
        full_paths = [os.path.join(folder_path, f) for f in text_files]
        sorted_file_paths = sorted(full_paths, key=lambda x: extract_leading_number_from_filename(os.path.basename(x)), reverse=not ascending)

        return sorted_file_paths
    except Exception as e:
        logging.error(f"retrieve_sorted_txt_files failed to retrieve and sort files from {folder_path}: {e}")
        logging.error(traceback.format_exc())

        return None

# Create Lists of str from files
def create_string_list_from_file_list(
    file_path_list: List[str]
) -> List[str]:
    """
        Create a list of strings from a list of file paths.

        Args:
            file_path_list (List[str]): A list of file paths.

        Returns:
            List[str]: A list of strings retrieved from the files.

        Raises:
            IOError: If there is an error opening or reading a file.
            ValueError: If the file content is unexpectedly empty or malformed.
    """
    string_list = []
    for file_path in file_path_list:
        try:
            file_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                if text:
                    string_list.append(text)
                else:
                    raise ValueError(f"File {file_name} is empty or contains only whitespace.")
        except Exception as e:
            logging.error(f"create_string_list_from_file_list failed to process file {file_name}: {e}")
            logging.error(traceback.format_exc())
            raise IOError(f"Error reading from {file_name}") from e

    return string_list

# Combine string lists
def concatenate_list_text_to_list_text(
    first_list: List[str],
    second_list: List[str],
    delimiter: str = f"\n{'-'*10}\n",
) -> List[str]:
    """
        Concatenates each element of the second list to the corresponding element of the first list,
        separated by the specified delimiter. Returns a new list with the concatenated text.

        Args:
            first_list (List[str]): The first list of strings.
            second_list (List[str]): The second list of strings.
            delimiter (str): The delimiter to separate the elements of the first and second lists.

        Returns:
            List[str]: A new list with the modified text.

        Raises:
            ValueError: If the input lists are not of the same length.
    """
    if not all(isinstance(i, str) for i in first_list) or not all(isinstance(i, str) for i in second_list):
        logging.error("Both input lists should contain only strings.")
        return None

    if len(first_list) != len(second_list):
        logging.error("Input lists must be of the same length.")
        return None
    
    concatenated_list = []
    for i in range(len(first_list)):
        combined_text = f"{first_list[i]}{delimiter}{second_list[i]}"
        concatenated_list.append(combined_text)
    
    return concatenated_list

# Validate response types
def evaluate_valid_response(
    response: str, 
    expected_type: type
) -> dict:
    """
        Evaluates and validates the response from the API.

        Args:
            response (str): The raw string response from the API.
            expected_type (type): The expected data type of the response after evaluation.

        Returns:
            dict: The evaluated response as a dictionary if it matches the expected type.

        Raises:
            SyntaxError: If there is a syntax error when evaluating the response.
            ValueError: If the evaluated response is not of the expected type.
            Exception: For any other unexpected errors during evaluation.

        Example:
            >>> response = "```python\n{'speaker_1': 'John', 'speaker_2': 'Jane'}\n```"
            >>> result = evaluate_and_validate_response(response, dict)
            >>> print(result)
            {'speaker_1': 'John', 'speaker_2': 'Jane'}
    """
    try:
        logging.info(f"Validating that response is {expected_type}")
        eval_response = eval(response)
        if isinstance(eval_response, expected_type):
            logging.info("Response successfully evaluated and validated.")
            return eval_response
        else:
            raise ValueError("Response is not of the expected type.")
    except SyntaxError as e:
        logging.error(f"Syntax error during evaluation: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during evaluation: {e}")
        raise

def evaluate_and_clean_valid_response(
    response: str, 
    expected_type: type
) -> dict:
    """
        Cleans and validates the response from the API.

        This function attempts to evaluate the response directly and validate it against the expected type.
        If the direct evaluation fails, it tries to clean the response by removing any code block markers
        (e.g., ```python) and then re-evaluates the cleaned response.

        Args:
            response (str): The raw string response from the API.
            expected_type (type): The expected data type of the response after evaluation.

        Returns:
            dict: The evaluated response as a dictionary if it matches the expected type.

        Raises:
            ValueError: If the response cannot be evaluated and validated after cleaning.

        Example:
            >>> response = "```python\n{'speaker_1': 'John', 'speaker_2': 'Jane'}\n```"
            >>> result = clean_and_validate_response(response, dict)
            >>> print(result)
            {'speaker_1': 'John', 'speaker_2': 'Jane'}
    """

    try:
        return evaluate_valid_response(response, expected_type)
    except (SyntaxError, ValueError) as e:
        logging.error(f"Failed to validate evaluated response: {e}")
        logging.info(f"Try to clean response and re-evaluate...")
        cleaned_response = re.sub(r'```[\w]+', '', response).replace('```', '').strip()
        return evaluate_valid_response(cleaned_response, expected_type)

# Clean string outputs 
def convert_string_blocks_to_bullet_points(
    text: str, 
    bullet_marker: str = '-', 
    newline: str = '\n'
) -> str:
    """
        Converts blocks of text in a string to markdown bullet points.

        Args:
            text (str): The string containing the text.
            bullet_marker (str): The marker to use for bullet points. Default is '-'.
            newline (str): The newline character to use. Default is '\n'.

        Returns:
            str: The text with each block converted to a bullet point, or an empty string if input is empty.

        Examples:
            >>> convert_blocks_to_bullet_points("Hello\\n\\nWorld")
            '- Hello\\n- World'
    """
    if not text.strip():
        return ''

    split_blocks = text.split(newline + newline)
    bullet_points = [f"{bullet_marker} " + block.replace(newline, ' ').strip() for block in split_blocks]
    combined_blocks = newline.join(bullet_points)

    return combined_blocks

def replace_target_strings_in_file(
    file_path: str, 
    replacement_dict: dict
) -> None:
    """
    Replace target strings in a file based on a dictionary of replacements.

    Args:
        file_path (str): The path to the file to be modified.
        replacement_dict (dict): A dictionary where the keys are the target strings to be replaced and the values are the replacement strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there are issues reading or writing to the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = file.read()

        for old_string, new_string in replacement_dict.items():
            file_data = file_data.replace(old_string, new_string)

        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.write(file_data)
        os.replace(temp_file_path, file_path)

    except IOError as e:
        logging.error(f"replace_target_strings_in_file failed to replace strings in {file_path}: {e}")
        raise