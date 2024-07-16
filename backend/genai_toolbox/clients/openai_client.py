from openai import OpenAI

import os
import logging

logging.basicConfig(level=logging.INFO)

def openai_client():
    try:
        return OpenAI()
    except Exception as e:
        logging.error(f"Error getting OpenAI client: {e}")
        raise e