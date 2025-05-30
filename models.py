import os
import uuid
import requests
import subprocess
from transformers import pipeline
import imageio_ffmpeg

# 1) Ottieni il path al binario ffmpeg da imageio-ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Inizializza il pipeline di accent-classification
accent_pipe = pipeline(
    task="audio-classification",
    model="dima806/english_accents_classification",
    trust_remote_code=True  # nel caso il modello usi custom code
)

def download_video(url: str) -> str:
    """Scarica il video da URL in un file MP4 temporaneo."""
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    """
    Usa ffmpeg via subprocess per estrarre un WAV mono 16 kHz.
    Non serve MoviePy.
    """
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    cmd = [
        FFMPEG_EXE,
        "-y",               # overwrite se esiste
        "-i", video_path,   # input
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def predict_accent_from_url(url: str):
    """
    1) Scarica il MP4
    2) Estrai l’audio WAV via ffmpeg subprocess
    3) Passa il WAV al pipeline HF
    4) Restituisci (label, score), poi pulisci i temporanei
    """
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)

        result = accent_pipe(ap, top_k=1)[0]
        label = result["label"]
        score = float(result["score"])
        return label, score

    finally:
        for fpath in (vp, ap):
            if fpath and os.path.exists(fpath):
                os.remove(fpath)
