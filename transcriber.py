from moviepy.editor import *
import whisper
import time


def main():
    start_time = time.time()
    videoToAudio("video.mp4")
    audioToText("audio.wav")

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


main()
