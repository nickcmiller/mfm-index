from genai_toolbox.transcription.assemblyai_functions import replace_speakers_in_assemblyai_utterances, generate_assemblyai_transcript_with_speakers, identify_speakers
from genai_toolbox.download_sources.youtube_functions import generate_episode_summary
import json

ids_to_fix = [
    
]

with open("tmp/chunks_to_embed.json", "r") as f:
    episodes = json.load(f)

for episode in episodes:
    utterances = episode['utterances_dict']
    transcript = episode['utterances_dict']['transcript']

    summary = generate_episode_summary(
        title=episode['title'],
        description=episode['description'],
        feed_keywords=episode['feed_keywords'],
        feed_title=episode['feed_title'],
        feed_description=episode['feed_description'],
    )
    print(f"{'*'*20}\n Summary:\n {summary} \n{'*'*20}")

    replaced_utterances = replace_speakers_in_assemblyai_utterances(
        utterances=utterances, 
        summary=summary, 
        output_dir_name="test"
    )

    episode['replaced_dict'] = replaced_utterances

    print(f"{'*'*20}\n Replaced:\n {json.dumps(episode['replaced_dict']['transcribed_utterances'][0], indent=4)} \n{'*'*20}")

with open("tmp/test_chunks.json", "w") as f:
    json.dump(episodes, f, indent=4)