import os
import uuid
import requests
import imageio_ffmpeg       # ← import first
# point moviepy at the ffmpeg binary bundled in imageio-ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

from moviepy.editor import VideoFileClip
from transformers import pipeline

# initialize the accent-classification pipeline
accent_pipe = pipeline(
    task="audio-classification",
    model="dima806/english_accents_classification",
)

def download_video(url: str) -> str:
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    # context manager ensures file handles are closed
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(
            wav_path,
            fps=16000,
            codec="pcm_s16le"
        )
    return wav_path

def predict_accent_from_url(url: str):
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)

        result = accent_pipe(ap, top_k=1)[0]
        return result["label"], float(result["score"])

    finally:
        for path in (vp, ap):
            if path and os.path.exists(path):
                os.remove(path)
