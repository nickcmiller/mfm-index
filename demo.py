from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torch
import os

load_dotenv()
# Need to accept the pyannote terms of service on Hugging Face website
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_ACCESS_TOKEN)

# Check if pipeline is successfully created
if pipeline is None:
    raise ValueError("Failed to load the pipeline. Check your Hugging Face access token.")

# send pipeline to GPU if CUDA is available
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("Former OPD Chief LeRonne Armstrong announces city council run.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
