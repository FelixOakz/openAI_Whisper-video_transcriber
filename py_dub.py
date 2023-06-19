import speech_recognition as sr
from pydub import AudioSegment
import os

# Load the video file
video = AudioSegment.from_file("video.mp4", format="mp4")
duration = int(video.duration_seconds * 1000)  # Convert to milliseconds

# Split the video into 10-second segments
segment_length = 10 * 1000  # 10 seconds
segments = range(0, duration, segment_length)

for i, start_time in enumerate(segments):
    # Extract segment from video
    end_time = start_time + segment_length
    segment = video[start_time:end_time]

    # Export segment as WAV file
    segment.export(f"audio_segment_{i}.wav", format="wav")

    # Transcribe the segment
    r = sr.Recognizer()
    with sr.AudioFile(f"audio_segment_{i}.wav") as source:
        audio_text = r.record(source)
    
    try:
        text = r.recognize_google(audio_text, language='en-US')
        # Append segment's transcript to the final transcript
        with open("transcription.txt", "a") as file:
            file.write(text + " ")
    except sr.UnknownValueError:
        print(f"No speech detected in segment {i}")

# Open the transcript file for editing by the user
os.system("start transcription.txt")
