from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_episode_summary
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_utterances, replace_speakers_in_assemblyai_utterances
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
from config import PODCAST_CONFIG

from typing import List, Dict, Optional
import os
import traceback
import logging

import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO)

async def download_and_transcribe_multiple_episodes_by_date(
    feed_url: str,
    start_date: str,
    end_date: str,
    audio_dir_name: Optional[str] = None,
    utterances_dir_name: Optional[str] = None,
    utterances_replaced_dir_name: Optional[str] = None,
    max_concurrent_tasks: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, str]]:
    feed_entries = return_entries_by_date(feed_url, start_date, end_date)
    logging.info(f"Downloading {len(feed_entries)} episodes")

    semaphore = Semaphore(max_concurrent_tasks)

    async def process_entry(entry: Dict[str, str], pbar: tqdm) -> Dict[str, str]:
        async with semaphore:
            async def retry_stage(stage_name, stage_func, *args):
                for attempt in range(max_retries):
                    try:
                        pbar.set_description(f"Processing {entry['title'][:30]}...")
                        pbar.set_postfix({"stage": f"{stage_name} in progress"})
                        result = await asyncio.to_thread(stage_func, *args)
                        pbar.set_postfix({"stage": f"{stage_name} completed"})
                        return result
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logging.warning(f"Error in {stage_name} for {entry['title']}, attempt {attempt + 1}: {str(e)}")
                            logging.debug(f"Traceback: {traceback.format_exc()}")
                            pbar.set_postfix({"stage": f"{stage_name} retry {attempt + 1}/{max_retries}"})
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                        else:
                            logging.error(f"Failed {stage_name} for {entry['title']} after {max_retries} attempts: {str(e)}")
                            logging.debug(f"Traceback: {traceback.format_exc()}")
                            pbar.set_postfix({"stage": f"{stage_name} failed"})
                            raise

            try:
                entry['audio_file_path'] = await retry_stage(
                    "Audio download", download_podcast_audio,
                    entry['url'], entry['title'], audio_dir_name
                )
                
                entry['audio_summary'] = await retry_stage(
                    "Summary generation", generate_episode_summary,
                    entry['summary'], entry['feed_summary']
                )
                
                entry['utterances_dict'] = await retry_stage(
                    "Utterances generation", generate_assemblyai_utterances,
                    entry['audio_file_path'], utterances_dir_name
                )
                
                entry['replaced_dict'] = await retry_stage(
                    "Utterances replacement", replace_speakers_in_assemblyai_utterances,
                    entry['utterances_dict'], entry['audio_summary'], utterances_replaced_dir_name
                )

                if os.path.exists(entry['audio_file_path']):
                    os.remove(entry['audio_file_path'])
                    logging.info(f"Removed audio file: {entry['audio_file_path']}")
                else:
                    logging.warning(f"Audio file not found for cleanup: {entry['audio_file_path']}")
                
                pbar.update(1)
                return entry
            except Exception:
                pbar.set_postfix({"stage": "failed"})
                pbar.update(1)
                return None

    with tqdm(total=len(feed_entries), desc="Processing episodes") as pbar:
        tasks = [process_entry(entry, pbar) for entry in feed_entries]
        updated_entries = await asyncio.gather(*tasks)

    successful_entries = [entry for entry in updated_entries if entry is not None]
    logging.info(f"Successfully processed {len(successful_entries)} out of {len(feed_entries)} episodes")
    return successful_entries

def process_podcast_feed(
    podcast_config: dict
    
):
    new_episodes = asyncio.run(download_and_transcribe_multiple_episodes_by_date(
        feed_url=podcast_config['feed_url'],
        start_date=podcast_config['start_date'],
        end_date=podcast_config['end_date'],
        audio_dir_name=podcast_config['audio_dir_name'],
    ))

    write_to_file(
        content=new_episodes, 
        file=podcast_config['output_file_name'], 
        output_dir_name=podcast_config['output_dir_name']
    )

    return new_episodes


if __name__ == "__main__":
    import json
    new_episodes = process_podcast_feed()
    print(json.dumps(new_episodes[0], indent=4))