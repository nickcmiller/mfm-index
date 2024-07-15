import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

chat_state = []

# Initialize or load messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "What question would you like to ask about Dithering?"}]

# Chat input
prompt = st.chat_input("Say something")

# Handle new message
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    
    try:
        # Call the backend API
        response = requests.post(
            f"{BACKEND_URL}/ask_question",
            json={
                "question": prompt,
                "chat_state": st.session_state['messages'],
                "table_name": TABLE_NAME
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Append the response to the chat history
        st.session_state['messages'].append({"role": "assistant", "content": result['llm_response']})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.error(error_message)
        st.session_state['messages'].append({"role": "assistant", "content": error_message})

# Function to display all messages
def display_messages():
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Display messages
display_messages()