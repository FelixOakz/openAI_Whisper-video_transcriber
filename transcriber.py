import datetime
import wave
import contextlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

# Upload audio file
path = input("Enter the path to the audio file: ")

num_speakers = 2

language = 'any'

model_size = 'medium'

model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'

# Set the device to CPU
device = torch.device("cpu")

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=device)

with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

audio = Audio()

def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)

    # Convert waveform to mono if it has multiple channels
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    return embedding_model(waveform[None])

model = whisper.load_model(model_size)

result = model.transcribe(path)
segments = result["segments"]

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
    return datetime.timedelta(seconds=round(secs))

transcript_file = "transcript.txt"

with open(transcript_file, "w") as f:
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')
