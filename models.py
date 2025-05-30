import os
import uuid
import requests
import subprocess

import imageio_ffmpeg
import torchaudio
from transformers import pipeline

# 1) Get the ffmpeg binary bundled with imageio-ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Init the HF audio-classification pipeline
accent_pipe = pipeline(
    task="audio-classification",
    model="dima806/english_accents_classification",
)

def download_video(url: str) -> str:
    """Download the video from URL to a temporary .mp4 file."""
    mp4_path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(mp4_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return mp4_path

def extract_audio(video_path: str) -> str:
    """
    Use subprocess + imageio-ffmpeg's ffmpeg to extract a mono 16 kHz WAV.
    """
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    cmd = [
        FFMPEG_EXE,
        "-y",               # overwrite
        "-i", video_path,   # input file
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def predict_accent_from_url(url: str):
    """
    1) Download MP4
    2) Extract WAV via ffmpeg subprocess
    3) Load WAV via torchaudio → numpy array
    4) Run HF pipeline on the raw array
    5) Cleanup temp files
    """
    mp4_file = wav_file = None
    try:
        mp4_file = download_video(url)
        wav_file = extract_audio(mp4_file)

        # Load with torchaudio (no ffmpeg needed here)
        waveform, sr = torchaudio.load(wav_file)   # waveform: Tensor [chan, time]
        audio = waveform.squeeze().numpy()         # shape [time] for mono

        # Pass the numpy array + sampling rate
        result = accent_pipe(audio, sampling_rate=sr, top_k=1)[0]
        return result["label"], float(result["score"])

    finally:
        for p in (mp4_file, wav_file):
            if p and os.path.exists(p):
                os.remove(p)

