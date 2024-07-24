import logging
import os
import json
import re

import streamlit as st
from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests

load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
BACKEND_URL = os.getenv("BACKEND_URL")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize or load messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "I have access to transcripts of the My First Million podcast. What would you like to know?"}]

def get_id_token(audience):
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    auth_req = Request()
    id_token_credentials = id_token.fetch_id_token(auth_req, audience)
    return id_token_credentials

def make_authorized_request(url, method='GET', **kwargs):
    audience = url
    token = get_id_token(audience)
    headers = kwargs.pop('headers', {})
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request(method, url, headers=headers, **kwargs)
    return response

def clean_text(text):
    # Decode Unicode escape sequences if needed
    try:
        text = text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        pass  # If decoding fails, skip this step

    # Replace newline escape sequences with actual newlines
    text = text.replace('\\n', '\n')

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove Markdown formatting (asterisks and underscores)
    text = re.sub(r'(\*|_){1,2}(?=\S)(.+?)(?<=\S)\1', r'\2', text)

    # Escape special characters for Streamlit, excluding $
    special_chars = "\\{}[]()#+-.!_*&"
    for char in special_chars:
        text = text.replace(char, f"\\{char}")

    return text

# Chat input
prompt = st.chat_input("Say something")

# Handle new message
if prompt:
    st.session_state['messages'].append({"role": "user", "content": clean_text(prompt)})
    
    try:
        # Call the backend API with authenticated request
        full_url = f"{BACKEND_URL}/ask_question"
        
        payload = {
            "question": prompt,
            "chat_state": st.session_state['messages'],
            "table_name": TABLE_NAME
        }

        logger.info(f"Sending request to backend: {full_url}\nPayload: {json.dumps(payload, indent=2)}")

        response = make_authorized_request(full_url, method='POST', json=payload)
        
        logger.info(f"Response status code: {response.status_code}\nResponse headers: {response.headers}")
        
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Raw response from backend: {json.dumps(result, indent=2)}")
        
        if 'llm_response' in result:
            st.session_state['messages'].append({"role": "assistant", "content": clean_text(result['llm_response'])})
        else:
            logger.error(f"Unexpected response format: {result}")
            st.error("Received an unexpected response format from the backend.")
    except requests.exceptions.RequestException as e:
        error_message = f"An error occurred: {str(e)}"
        if hasattr(e, 'response'):
            error_message += f"\nResponse: {e.response.text}"
            error_message += f"\nStatus Code: {e.response.status_code}"
            error_message += f"\nHeaders: {e.response.headers}"
        st.error(error_message)
        logger.error(error_message)

# Function to display all messages
def display_messages():
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(clean_text(message["content"]))

# Display messages
display_messages()