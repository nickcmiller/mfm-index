import logging
import asyncio
from tqdm.asyncio import tqdm
from genai_toolbox.transcription.assemblyai_functions import replace_speakers_in_assemblyai_utterances, summarize_transcript
from genai_toolbox.helper_functions.string_helpers import retrieve_file, write_to_file
from gcs_functions import process_and_upload_entries

async def process_utterances(config):
    utterances_data = retrieve_file(
        file=config['input_file'],
        dir_name=config['input_dir']
    )
    
    logging.info(f"Retrieved {len(utterances_data)} entries to process")

    async def process_entry(entry):
        try:
            result = await asyncio.to_thread(
                replace_speakers_in_assemblyai_utterances,
                entry['utterances_dict'],
                entry['summary_generation'],
            )
            entry['replaced_dict'] = result
            logging.info(f"Successfully processed entry: {entry['title']}")
            return entry
        except Exception as e:
            logging.error(f"Error processing utterances for {entry['title']}: {str(e)}")
            return None

    tasks = [process_entry(entry) for entry in utterances_data]
    successful_entries = []
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing utterances"):
        result = await f
        if result is not None:
            successful_entries.append(result)
    
    logging.info(f"Successfully processed {len(successful_entries)} out of {len(utterances_data)} entries")
    
    if successful_entries:
        logging.info(f"Writing {len(successful_entries)} entries to file")
        await asyncio.to_thread(
            write_to_file,
            content=successful_entries,
            file=config['output_file_name'],
            output_dir_name=config['output_dir_name']
        )
        logging.info(f"Successfully written {len(successful_entries)} entries to {config['output_file_name']}")
    else:
        logging.warning("No successful entries to write to file")

    if config['upload_to_gcs']:
        await asyncio.to_thread(
            process_and_upload_entries,
            successful_entries,
            config['gcs_bucket_name'],
        )

    return successful_entries

def replace_speakers(config):
    return asyncio.run(process_utterances(config))

if __name__ == "__main__":
    from dotenv import load_dotenv
    import json
    load_dotenv()

    speaker_replaced_utterances = retrieve_file(
        file="speaker_replaced_utterances.json",
        dir_name="tmp"
    )
    transcript = speaker_replaced_utterances[0]['replaced_dict']['transcript']

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
                    "provider": "groq",
                    "model": "llama3.1-70b"
                },
                {
                    "provider": "openai",
                    "model": "4o-mini"
                },
            ],
            "system_instructions": system_instructions
        }
    ]
    
    summary = summarize_transcript(
        prompt_list=prompt_list,
        transcript=transcript,
    )
    

    # print(json.dumps(summary['revision_dict'], indent=4))
    for k in list(summary['revision_dict'].keys()):
        print(f"{'*' * 50}\n{k}\n{'*' * 50}\n")
        print(summary['revision_dict'][k][0])