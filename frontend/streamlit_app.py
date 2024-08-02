import logging
import os
import json
import re
import time
import requests
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token


load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_backend_api(
    backend_url: str,
    prompt: str,
    chat_state: list[dict],
    table_name: str
) -> Generator[str, None, None]:
    full_url = f"{backend_url}/ask_question"
    
    payload = {
        "question": prompt,
        "chat_state": chat_state,
        "table_name": table_name
    }

    logger.info(f"Sending request to backend: {full_url}\nPayload: {json.dumps(payload, indent=2)}")
    
    try:
        response = _make_authorized_request(
            backend_url=backend_url,
            url=full_url,
            method='POST',
            json=payload,
            stream=True
        )
        
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

def _make_authorized_request(
    backend_url: str,
    url: str,
    method: str = 'GET',
    **kwargs
) -> requests.Response:
    if _is_local_environment(backend_url):
        return requests.request(method, url, **kwargs)
    else:
        audience = url
        token = _get_id_token(audience)
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        return requests.request(method, url, headers=headers, **kwargs)

def _is_local_environment(
    backend_url: str
) -> bool:
    logger.info(f"RUNNING IN LOCAL ENVIRONMENT: {backend_url}")
    return backend_url.startswith("http://localhost") or backend_url.startswith("http://127.0.0.1")

def _get_id_token(
    audience: str
) -> str:
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    auth_req = Request()
    id_token_credentials = id_token.fetch_id_token(auth_req, audience)
    return id_token_credentials

def display_messages(
    messages: list[dict]
) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(_clean_text(message["content"]))

def _clean_text(
    text: str
) -> str:
    # Replace newline escape sequences with actual newlines
    text = text.replace('\\n', '\n')

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove Markdown formatting (asterisks and underscores)
    # text = re.sub(r'(\*|_){1,2}(?=\S)(.+?)(?<=\S)\1', r'\2', text)

    # Escape special characters for Streamlit, excluding $ and backslashes in URLs
    special_chars = "\\{}[]()#+-.!_*&"
    for char in special_chars:
        if char != '\\':
            text = text.replace(char, f"\\{char}")

    # Handle backslashes in URLs
    text = re.sub(r'\\(?![\\n])', '', text)

    # Escape dollar signs
    text = text.replace('$', '\\$')

    return text

def handle_new_message(
    prompt: str,
    backend_url: str,
    table_name: str
) -> None:
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show loading spinner while waiting for the response
    with st.spinner("Generating response..."):
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

        try:
            for token in call_backend_api(
                backend_url=backend_url,
                prompt=prompt,
                chat_state=st.session_state['messages'],
                table_name=table_name
            ):
                full_response += _clean_text(token)
                # Use markdown to display the streaming response
                message_placeholder.markdown(full_response + "▌")
            
            # Final update to remove cursor
            message_placeholder.markdown(full_response)

            # Add assistant's message to chat history
            st.session_state['messages'].append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            _handle_request_exception(e)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")


# Function to handle request exceptions
def _handle_request_exception(e: requests.exceptions.RequestException) -> None:
    error_message = f"An error occurred: {str(e)}"
    if hasattr(e, 'response'):
        try:
            error_details = e.response.json()  # Try to parse JSON response
            error_message += f"\nError details: {json.dumps(error_details, indent=2)}"
        except json.JSONDecodeError:
            error_message += f"\nResponse text: {e.response.text}"
        error_message += f"\nStatus Code: {e.response.status_code}"
        error_message += f"\nHeaders: {dict(e.response.headers)}"
    elif hasattr(e, 'request'):
        error_message += f"\nRequest URL: {e.request.url}"
        error_message += f"\nRequest Method: {e.request.method}"
    st.error(error_message)
    logger.error(error_message, exc_info=True)

TABLE_NAME = os.getenv("TABLE_NAME")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
 
st.title("MFM Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have access to transcripts of the My First Million podcast. Ask me some questions."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Handle new message
    handle_new_message(
        prompt=prompt,
        backend_url=BACKEND_URL,
        table_name=TABLE_NAME
    )