from groq import Groq

import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')


def groq_text_response(
    messages: List[dict],
    model: str="llama3-8b-8192"
) -> str:
    client = Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )

    text_response = chat_completion.choices[0].message.content

    return text_response

