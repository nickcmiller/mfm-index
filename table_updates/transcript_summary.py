from genai_toolbox.text_prompting.prompt_chaining import revise_string_with_prompt_list

def summarize_transcript(
    transcript: str
) -> str:
    system_instructions = """
        Adhere to the following formatting rules when creating the outline:
            - Start with a top-level header (###) for the text title.
            - Use up to five header levels for organization.
            - Use hyphens (-) exclusively for bullet points.
            - Never use other symbols (e.g., •, ‣, or ■) or characters (+, *, etc.) for bullets.
            - Indent bullet points according to the hierarchy of the Markdown outline.
            - Always indent subheaders under headers.
            - Use consistent indentation for hierarchy and clarity.
            - Never usng an introduction sentence (e.g., 'Here is the...') before the outline. 
            - Use header levels effectively to organize content within the outline.
            - Only use bullets and headers for formatting. Do not use any other type of formatting. 
    """

    starting_prompt = """
        Craft a long outline reflecting the main points of a transcript using Markdown formatting. Adhere to these rules:
            - Identify the transcript title in the first header.
            - Identify the speakers who participated in the conversation. 
            - Under each header, thoroughly summarize transcript's topics, key terms, concepts, and themes in detail. 
            - Under the same headers, list pertinent questions raised by the transcript.
        The aim is to organize the transcript's essence into relevant, detailed bullet points and questions.

        Transcript:
        {}
    """

    second_prompt = f"""
        Using the text provided in the transcript, increase the content of the outline while maintaining original Markdown formatting. In your responses, adhere to these rules:
            - Expand each bullet point with detailed explanations and insights based on the transcript's content.
            - Answer the questions posed in the outline.
            - When appropriate, define terms or concepts.
            - Use the transcript for evidence and context.
        Aim for detailed and informative responses that enhance understanding of the subject matter.

        Outline:
        {{}}
    """

    prompt_list = [
        {
            "instructions": starting_prompt, 
            "model_order": [
                {
                    "provider": "openai",
                    "model": "4o-mini"
                },
                {
                    "provider": "groq",
                    "model": "llama3.1-8b"
                },
                
            ],
            "system_instructions": system_instructions
        },
        {
            "instructions": second_prompt, 
            "model_order": [
                {
                    "provider": "openai",
                    "model": "4o"
                },
                {
                    "provider": "anthropic",
                    "model": "sonnet"
                },
                {
                    "provider": "groq",
                    "model": "llama3.1-70b"
                },
            ],
            "system_instructions": system_instructions
        }
    ]

    summary = revise_string_with_prompt_list(
        string=transcript,
        prompt_list=prompt_list,
        concatenation_delimiter=f"\n{'-'*3}\nTranscript:\n{'-'*3}\n",
    )

    return summary['modified_string']

def convert_summary_to_utterance(
    summary: str,
    video_id: str,
    episode_title: str,
    feed_title: str,
    feed_regex: str,
    episode_regex: str,
    episode_date: str
) -> str:

    summary_id = f"summary {feed_regex} {episode_regex} ".replace(' ', '-')
    youtube_link = f"https://www.youtube.com/watch?v={video_id}"
    
    return {
        "speakers": ["AI Generated Summary"],
        "text": summary,
        "start_mins": "0:00",
        "end_mins": "0:00",
        "start_ms": 0,
        "end_ms": 0,
        "title": episode_title,
        "publisher": feed_title,
        "date_published": episode_date,
        "youtube_link": youtube_link,
        "id": summary_id
    }

if __name__ == "__main__":
    from genai_toolbox.helper_functions.string_helpers import retrieve_file
    import json
    from dotenv import load_dotenv
    load_dotenv()

    speaker_replaced_utterances = retrieve_file(
        file="speaker_replaced_utterances.json",
        dir_name="tmp"
    )
    transcript = speaker_replaced_utterances[0]['replaced_dict']['transcript']

    summary = summarize_transcript(transcript)