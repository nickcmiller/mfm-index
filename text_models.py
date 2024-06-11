from groq import Groq
from openai import OpenAI
    

from typing import List
import logging
import traceback
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')


def groq_text_response(
    prompt: str,
    history_messages: List[dict] = [],
    model: str = "llama3-8b-8192"
) -> str:
    """
        This function gets a text response from Groq. 

        Arguments:
            prompt: str - The prompt to send to the model.
            history_messages: List[dict] - The messages to send to the model.
            model: str - The model to use.

        Returns:
            str - The text response from the model.

        Example:
            message_list = [
                {
                    "role": "user",
                    "content": "Explain the importance of fast language models",
                }
            ]
            >>> groq_text_response(message_list)

            "Fast language models are important because they can be used to generate text in real-time."
    """
    client = Groq()

    messages = history_messages
    messages.append({
        "role": "user",
        "content": prompt,
    })
    logging.info(f"Messages Input Count: {len(messages)}")

    try:    
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )
        logging.info(f"Completion Usage: {chat_completion.usage}")
        text_response = chat_completion.choices[0].message.content
        return text_response
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        raise e

def openai_text_response(
    prompt: str,
    history_messages: List[dict] = [],
    model: str = "gpt-4o"
) -> str:
    client = OpenAI()

    messages = history_messages
    messages.append({
        "role": "user",
        "content": prompt,
    })
    logging.info(f"Messages Input Count: {len(messages)}")

    try:    
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )
        logging.info(f"Completion Usage: {chat_completion.usage}")
        text_response = chat_completion.choices[0].message.content
        return text_response
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        raise e