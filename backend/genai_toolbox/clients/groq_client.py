from groq import Groq

import os
import logging

logging.basicConfig(level=logging.INFO)

def groq_client():
    try:
        return Groq()
    except Exception as e:
        logging.error(f"Error getting Groq client: {e}")
        raise e