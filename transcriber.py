from moviepy.editor import *
import whisper
import time
from pyAudioAnalysis import audioSegmentation

def main():
    start_time = time.time()
    videoToAudio("video.mp4")
    audioToText("audio.wav")
    diarizeAudio("audio.wav")

    elapsed_time = time.time() - start_time
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

def audioToText(audio_path):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    print(result["text"])

def videoToAudio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("audio.wav", codec="pcm_s16le")

def diarizeAudio(audio_path):
    [segments, labels] = audioSegmentation.speaker_diarization(audio_path)
    print("Speaker Diarization Results:")
    for seg, label in zip(segments, labels):
        print("Segment: {:.2f} - {:.2f}, Label: {}".format(seg[0], seg[1], label))

main()
