import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import streamlit as st
from query_handler import question_with_chat_state
import os
from dotenv import load_dotenv
load_dotenv()

table_name = os.getenv("TABLE_NAME")
print(f"Table name: {table_name}")

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
        # Call question_with_chat_state with the new prompt
        response = question_with_chat_state(
            question=prompt,
            chat_state=st.session_state['messages'],
            table_name=table_name,
        )
        
        # Append the response to the chat history
        st.session_state['messages'].append({"role": "assistant", "content": response['llm_response']})
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