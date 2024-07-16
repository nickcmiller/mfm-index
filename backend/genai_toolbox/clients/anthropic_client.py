from anthropic import Anthropic

import os
import logging

logging.basicConfig(level=logging.INFO)

def anthropic_client():
    try:
        return Anthropic()
    except Exception as e:
        logging.error(f"Error getting Anthropic client: {e}")
        raise e