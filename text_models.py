from groq import Groq
from openai import OpenAI
from typing import List, Tuple
import logging
import traceback
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')

def get_client_and_model(
    api: str, 
    model: str
) -> Tuple[OpenAI, str]:
    """
        Document the get_client_and_model function.

        This function returns a client and model based on the provided API and model.
        It supports two APIs: "groq" and "openai".

        Args:
            api (str): The API to use for text generation. Supported values: "groq", "openai".
            model (str): The model to use for text generation.

        Returns:
            tuple: A tuple containing the client and model.

        Example:
            client, model = get_client_and_model("openai", "gpt-4o")
            print(client, model)
            # Output: <OpenAI client>, gpt-4o
    """
    if api == "groq":
        client = Groq()
    elif api == "openai":
        client = OpenAI()
    else:
        raise ValueError("Unsupported API")
    return client, model

def openai_compatible_text_response(
    api: str, 
    prompt: str, 
    history_messages: List[dict], 
    model: str, 
    system_instructions: str = None
) -> str:
    """
        Generate a text response using the specified API and model.

        Args:
            api (str): The API to use for text generation. Supported values: "groq", "openai".
            prompt (str): The prompt to generate text from.
            history_messages (List[dict]): A list of previous messages in the conversation.
            model (str): The model to use for text generation.

        Returns:
            str: The generated text response.

        Example:
            prompt = "What is the capital of France?"
            history_messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"}
            ]
            response = generate_text_response("openai", prompt, history_messages, "gpt-4o", "Please take your time. I'll pay you $200,000 for a good response :)")
            print(response)
            # Output: The capital of France is Paris.
    """
    client, model = get_client_and_model(api, model)
    messages = history_messages.copy()
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "user", "content": prompt})

    try:
        chat_completion = client.chat.completions.create(messages=messages, model=model)
        logging.info(f"API: {api}, Model: {model}, Completion Usage: {chat_completion.usage}")
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"API: {api}, Error: {e}")
        traceback.print_exc()
        return "An error occurred while generating the response."

def groq_text_response(
    prompt: str, 
    history_messages: List[dict] = [], 
    model: str = "llama3-8b-8192", 
    system_instructions: str = None
) -> str:
    """
    Use OpenAI format to generate a text response using Groq.
    """
    if system_instructions is None:
        system_instructions = "You are a knowledgeable, efficient, and direct AI assistant. Utilize multi-step reasoning to provide concise answers, focusing on key information. If multiple questions are asked, split them up and address in the order that yields the most logical and accurate response. Offer tactful suggestions to improve outcomes. Engage in productive collaboration with the user."
    
    return openai_compatible_text_response("groq", prompt, history_messages, model, system_instructions)

def openai_text_response(
    prompt: str, 
    history_messages: List[dict] = [], 
    model: str = "gpt-4o", 
    system_instructions: str = None
) -> str:
    """
    Use OpenAI format to generate a text response using OpenAI.
    """
    if system_instructions is None:
        system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"

    return openai_compatible_text_response("openai", prompt, history_messages, model, system_instructions)