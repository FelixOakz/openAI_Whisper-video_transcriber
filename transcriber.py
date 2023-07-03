from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import whisper
import time

def main():
    start_time = time.time()
    audio_path = videoToAudio("video.mp4")
    audioToText(audio_path)
    whisper_transcription(audio_path)

    elapsed_time = time.time() - start_time
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def whisper_transcription(audio_path):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    print("Whisper transcription:")
    print(result["text"])


def audioToText(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_path)

    print("Pyannote speaker diarization:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


def videoToAudio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = "audio.wav"
    audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


main()
