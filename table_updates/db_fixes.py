from genai_toolbox.transcription.assemblyai_functions import replace_speakers_in_assemblyai_utterances, generate_assemblyai_transcript_with_speakers, identify_speakers
from genai_toolbox.download_sources.youtube_functions import generate_episode_summary
from genai_toolbox.helper_functions.string_helpers import write_to_file
from gcs_functions import retrieve_entries_by_date_range, retrieve_entries_by_id, filter_index_by_date_range, retrieve_index_list_from_gcs, process_and_upload_entries
import json
from copy import deepcopy
from typing import List, Dict

BUCKET_NAME = "aai_utterances_json"

def replace_episode_speakers(
        entries: List[Dict]
) -> List[Dict]:
    """
        Replaces speakers in the provided episode entries.

        This function takes a list of episode entries, each containing a dictionary of utterances,
        and processes them to replace speaker labels based on the generated summary of the episode.
        It utilizes the `replace_speakers_in_assemblyai_utterances` function to perform the replacement
        and generates a summary using the `generate_episode_summary` function.

        Args:
            entries (List[Dict]): A list of dictionaries where each dictionary represents an episode entry
                                containing metadata and utterances.

        Returns:
            List[Dict]: A list of dictionaries with updated speaker labels and generated summaries.
    """
    fixed_entries = []
    for entry in entries:
        new_entry = deepcopy(entry)
        print(new_entry.keys())

        utterances = new_entry['utterances_dict']
        transcript = utterances['transcript']

        summary = generate_episode_summary(
            title=new_entry['title'],
            description=new_entry['description'],
            feed_keywords=new_entry['feed_keywords'],
            feed_title=new_entry['feed_title'],
            feed_description=new_entry['feed_description'],
        )
        print(f"{'*'*20}\n Summary:\n {summary} \n{'*'*20}")

        replaced_utterances = replace_speakers_in_assemblyai_utterances(
            utterances=utterances,
            summary=summary,
            output_dir_name="fixed_replacements"
        )
        new_entry['summary'] = summary
        new_entry['replaced_dict'] = replaced_utterances
        print(f"{'*'*20}\n Old Entry:\n {json.dumps(entry['replaced_dict']['transcribed_utterances'][:3], indent=4)} \n{'*'*20}")
        print(f"{'*'*20}\n Replaced:\n {json.dumps(new_entry['replaced_dict']['transcribed_utterances'][:3], indent=4)} \n{'*'*20}")

        fixed_entries.append(new_entry)

    return fixed_entries


def entries_main():
    #Make sure the booleans in youtube_config.py are set correctly
    # ids_to_fix = ["X0BCxa3V67M"]
    
    # entries = retrieve_entries_by_id(
    #     bucket_name="aai_utterances_json",
    #     video_ids=ids_to_fix
    # )

    entries = retrieve_entries_by_date_range(
        bucket_name=BUCKET_NAME,
        start_date="2024-02-1",
        end_date="2024-05-1"
    )

    return entries
if __name__ == "__main__":
    import time
    import json

    start_time = time.time()
    entries = entries_main()
    print(len(entries))
    print(f"Time taken: {time.time() - start_time} seconds")

    print(f"Entries keys: {sorted(entries[0].keys())}\nLength: {len(entries[0].keys())}")

    write_to_file(
        content=entries,
        file="speaker_replaced_utterances.json",
        output_dir_name="tmp"
    )
