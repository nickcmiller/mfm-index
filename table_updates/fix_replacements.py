from genai_toolbox.transcription.assemblyai_functions import replace_speakers_in_assemblyai_utterances, generate_assemblyai_transcript_with_speakers, identify_speakers
from genai_toolbox.download_sources.youtube_functions import generate_episode_summary
from gcs_functions import retrieve_entries_by_id, filter_index_by_date_range, retrieve_index_list_from_gcs
import json
from copy import deepcopy
from typing import List, Dict

def replace_episode_speakers(
        entries: List[Dict]
) -> List[Dict]:
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


def main():
    #Make sure the booleans in youtube_config.py are set correctly
    ids_to_fix = ["X0BCxa3V67M"]
    
    entries = retrieve_entries_by_id(
        bucket_name="aai_utterances_json",
        video_ids=ids_to_fix
    )

    fixed_entries = replace_episode_speakers(entries)

    with open("tmp/speaker_replaced_utterances.json", "w") as f:
        json.dump(fixed_entries, f, indent=4)

if __name__ == "__main__":
    main()