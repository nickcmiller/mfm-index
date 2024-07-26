import logging
import asyncio
from tqdm.asyncio import tqdm
from genai_toolbox.transcription.assemblyai_functions import replace_speakers_in_assemblyai_utterances
from genai_toolbox.helper_functions.string_helpers import retrieve_file, write_to_file

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

    return successful_entries

def replace_speakers(config):
    return asyncio.run(process_utterances(config))