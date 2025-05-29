import requests
from moviepy.editor import VideoFileClip
import whisper
import torchaudio
import librosa

def download_video(url, path="video.mp4"):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return path

def extract_audio(video_path, audio_path="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, fps=16000)
    return audio_path

def transcribe_and_check_english(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["language"] == "en", result["text"]

def get_fluency_score(transcript, audio_path):
    word_count = len(transcript.split())
    duration = librosa.get_duration(path=audio_path)
    wpm = word_count / (duration / 60)
    score = min(100, int(wpm))  # cap a 100
    return score
