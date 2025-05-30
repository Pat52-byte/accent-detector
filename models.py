import os
import uuid
import requests
from moviepy.editor import VideoFileClip
from huggingface_hub import InferenceApi
import json

# 1) Inizializza l’API di Inference per l’accent-ID model
inference = InferenceApi(
    repo_id="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    task="audio-classification"
)

def download_video(url: str) -> str:
    """Scarica il video da URL e lo salva in un file temporaneo."""
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    """Estrae l’audio PCM a 16 kHz da un file video MP4."""
    wav = f"tmp_{uuid.uuid4()}.wav"
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(
            wav,
            fps=16000,
            codec="pcm_s16le"
        )
    return wav

def predict_accent_from_url(url: str):
    """
    1) Scarica il video da URL
    2) Estrae l’audio come WAV
    3) Chiama l’API HF con raw_response=True
    4) Parsifica il JSON e restituisce (label, score)
    5) Pulisce i file temporanei
    """
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)

        # Chiedi il raw response e parsalo in JSON
        resp = inference(ap, raw_response=True)
        results = json.loads(resp.content.decode("utf-8"))

        label = results[0]["label"]
        score = float(results[0]["score"])
        return label, score

    finally:
        # Rimuovi i file temporanei
        for f in (vp, ap):
            if f and os.path.exists(f):
                os.remove(f)


