from dotenv import load_dotenv

load_dotenv()

from genai_toolbox.text_prompting.model_calls import fallback_response

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant that writes poems."},
        {"role": "user", "content": "Write everything as if you were Eminem."},
        {"role": "assistant", "content": "I will write like the rapper Eminem."},
    ]
    prompt = "Write a poem on cats"
    model_order = [
        # {"provider": "perplexity", "model": "llama3.1-70b"},
        {"provider": "openai", "model": "4o-mini"}
    ]
    stream_response = fallback_response(
        prompt=prompt, 
        history_messages=messages, 
        model_order=model_order, 
        stream=True
    )
    for chunk in stream_response:
        print(chunk, end='', flush=True)

    normal_response = fallback_response(
        prompt=prompt, 
        history_messages=messages, 
        model_order=model_order, 
        stream=False
    )

    print(f"Normal: {normal_response}")
