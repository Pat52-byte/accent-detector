import os
import uuid
import requests
from moviepy.editor import VideoFileClip
from huggingface_hub import InferenceApi

# 1) Crea l’Inference client per il modello di accent ID
inference = InferenceApi(
    repo_id="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    task="audio-classification"
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
    wav = f"tmp_{uuid.uuid4()}.wav"
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(
            wav, fps=16000, codec="pcm_s16le"
        )
    return wav

def predict_accent_from_url(url: str):
    vp = ap = None
    try:
        # scarica il video e ne estrai l’audio
        vp = download_video(url)
        ap = extract_audio(vp)

        # 2) Chiama l’API HF, ottieni etichetta e score
        results = inference(ap)
        # results è una lista di dict, con chiavi "label" e "score"
        label = results[0]["label"]
        score = float(results[0]["score"])

        return label, score

    finally:
        # pulizia file temporanei
        for f in (vp, ap):
            if f and os.path.exists(f):
                os.remove(f)



