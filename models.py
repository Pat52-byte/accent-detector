import os
import uuid
import requests
from moviepy.editor import VideoFileClip
from transformers import pipeline

# 1) Initialize the pipeline with a model that ships its own preprocessor
accent_pipe = pipeline(
    task="audio-classification",
    model="dima806/english_accents_classification",
)

def download_video(url: str) -> str:
    """Download the video from URL to a temporary .mp4 file."""
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    """Extract 16 kHz WAV audio from the MP4."""
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(
            wav_path,
            fps=16000,
            codec="pcm_s16le"
        )
    return wav_path

def predict_accent_from_url(url: str):
    """
    1) Download video → MP4
    2) Extract audio → WAV
    3) Run the audio-classification pipeline (top-1)
    4) Return (label, score) and clean up
    """
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)

        # pipeline returns a list of dicts; we take the first
        result = accent_pipe(ap, top_k=1)[0]
        label = result["label"]
        score = float(result["score"])
        return label, score

    finally:
        # remove temp files if they exist
        for path in (vp, ap):
            if path and os.path.exists(path):
                os.remove(path)


