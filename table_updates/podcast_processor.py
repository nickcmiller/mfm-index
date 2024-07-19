from genai_toolbox.download_sources.podcast_functions import return_entries_by_date, download_podcast_audio, generate_episode_summary
from genai_toolbox.transcription.assemblyai_functions import generate_assemblyai_utterances, replace_speakers_in_assemblyai_utterances
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file

from typing import List, Dict, Optional, Any
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

    async def process_entry(
        entry: Dict[str, str], 
        pbar: tqdm
    ) -> Dict[str, str]:
        async with semaphore:
            try:
                entry = await process_stages(
                    entry, 
                    pbar, 
                    max_retries, 
                    retry_delay, 
                    audio_dir_name, 
                    utterances_dir_name, 
                    utterances_replaced_dir_name
                )
                await cleanup_audio_file(entry, pbar)
                pbar.update(1)
                return entry
            except Exception as e:
                logging.error(f"Error processing entry {entry['title']}: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                pbar.set_postfix({"stage": "failed"})
                pbar.update(1)
                return None

    with tqdm(total=len(feed_entries), desc="Processing episodes") as pbar:
        tasks = [process_entry(entry, pbar) for entry in feed_entries]
        updated_entries = await asyncio.gather(*tasks)

    successful_entries = [entry for entry in updated_entries if entry is not None]
    logging.info(f"Successfully processed {len(successful_entries)} out of {len(feed_entries)} episodes")
    return successful_entries

async def process_stages(
    entry: Dict[str, str], 
    pbar: tqdm, 
    max_retries: int, 
    retry_delay: float, 
    audio_dir_name: Optional[str], 
    utterances_dir_name: Optional[str], 
    utterances_replaced_dir_name: Optional[str]
) -> Dict[str, str]:
    stages = [
        ("Audio download", download_podcast_audio, entry['url'], entry['title'], audio_dir_name),
        ("Summary generation", generate_episode_summary, entry['summary'], entry['feed_summary'])
    ]

    for stage_name, stage_func, *args in stages:
        try:
            logging.info(f"Starting {stage_name} for entry {entry['title']}")
            result = await retry_stage(stage_name, stage_func, entry, pbar, max_retries, retry_delay, *args)
            entry[stage_name.lower().replace(' ', '_')] = result
            logging.info(f"Completed {stage_name} for entry {entry['title']}")
        except Exception as e:
            logging.error(f"Error in stage {stage_name} for entry {entry['title']}: {str(e)}")
            logging.error(f"Entry state: {entry}")
            raise

    # Add the stages that depend on the results of previous stages
    if 'audio_download' in entry:
        entry['audio_file_path'] = entry['audio_download']
        try:
            stage_name = "Utterances generation"
            logging.info(f"Starting {stage_name} for entry {entry['title']}")
            result = await retry_stage(
                stage_name, 
                generate_assemblyai_utterances, 
                entry, 
                pbar, 
                max_retries, 
                retry_delay, 
                entry['audio_download'], 
                utterances_dir_name
            )
            entry['utterances_dict'] = result
            logging.info(f"Completed {stage_name} for entry {entry['title']}")
        except Exception as e:
            logging.error(f"Error in stage {stage_name} for entry {entry['title']}: {str(e)}")
            logging.error(f"Entry state: {entry}")
            raise

    if 'utterances_dict' in entry:
        try:
            stage_name = "Utterances replacement"
            logging.info(f"Starting {stage_name} for entry {entry['title']}")
            result = await retry_stage(
                stage_name, 
                replace_speakers_in_assemblyai_utterances, 
                entry, 
                pbar, 
                max_retries, 
                retry_delay, 
                entry['utterances_dict'], 
                entry['summary_generation'], 
                utterances_replaced_dir_name
            )
            entry['replaced_dict'] = result
            logging.info(f"Completed {stage_name} for entry {entry['title']}")
        except Exception as e:
            logging.error(f"Error in stage {stage_name} for entry {entry['title']}: {str(e)}")
            logging.error(f"Entry state: {entry}")
            raise

    return entry

async def retry_stage(
    stage_name: str, 
    stage_func, 
    entry: Dict[str, str], 
    pbar: tqdm, 
    max_retries: int, 
    retry_delay: float, 
    *args
) -> Any:
    for attempt in range(max_retries):
        try:
            pbar.set_description(f"Processing {entry['title'][:30]}...")
            pbar.set_postfix({"stage": f"{stage_name} in progress"})
            result = await asyncio.to_thread(stage_func, *args)
            pbar.set_postfix({"stage": f"{stage_name} completed"})
            return result
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} failed for {stage_name} in entry {entry['title']}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                raise

def handle_retry_exception(
    stage_name: str, 
    entry: Dict[str, str], 
    attempt: int, 
    max_retries: int, 
    e: Exception, 
    pbar: tqdm
) -> None:
    if attempt < max_retries - 1:
        logging.warning(f"Error in {stage_name} for {entry['title']}, attempt {attempt + 1}: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        pbar.set_postfix({"stage": f"{stage_name} retry {attempt + 1}/{max_retries}"})
    else:
        logging.error(f"Failed {stage_name} for {entry['title']} after {max_retries} attempts: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        pbar.set_postfix({"stage": f"{stage_name} failed"})

async def cleanup_audio_file(
    entry: Dict[str, str], 
    pbar: tqdm, 
    audio_dir_name: Optional[str] = None, 
    utterances_dir_name: Optional[str] = None, 
    utterances_replaced_dir_name: Optional[str] = None
) -> None:
    if 'audio_file_path' in entry and os.path.exists(entry['audio_file_path']):
        os.remove(entry['audio_file_path'])
        logging.info(f"Removed audio file: {entry['audio_file_path']}")
    else:
        logging.warning(f"Audio file not found for cleanup: {entry.get('audio_file_path', 'Not set')}")

def process_podcast_feed(
    podcast_config: dict
) -> List[Dict[str, str]]:
    try:
        new_episodes = asyncio.run(download_and_transcribe_multiple_episodes_by_date(
            feed_url=podcast_config['feed_url'],
            start_date=podcast_config['start_date'],
            end_date=podcast_config['end_date'],
            audio_dir_name=podcast_config['audio_dir_name'],
        ))
        
        logging.info(f"Processed episodes: {len(new_episodes)}")
        
        if not new_episodes:
            logging.warning("No episodes were successfully processed")
            return []

        # Filter out None values from new_episodes
        new_episodes = [episode for episode in new_episodes if episode is not None]

        if new_episodes:
            write_to_file(
                content=new_episodes, 
                file=podcast_config['output_file_name'], 
                output_dir_name=podcast_config['output_dir_name']
            )
        else:
            logging.warning("No valid episodes to write to file")

        return new_episodes
    except Exception as e:
        logging.error(f"Error in process_podcast_feed: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return []