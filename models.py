import os
import uuid
import requests
import subprocess
import wave

import numpy as np
import torch
import imageio_ffmpeg
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# 1) Path al binario FFmpeg bundlato in imageio-ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Carica una volta il feature extractor e il modello
MODEL_ID = "dima806/english_accents_classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def download_video(url: str) -> str:
    """Scarica il video da URL in un file MP4 temporaneo."""
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True); resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    """Estrai WAV mono 16 kHz via FFmpeg subprocess."""
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def load_wav(wav_path: str):
    """
    Carica un WAV a 16 kHz usando il modulo wave + numpy.
    Restituisce (numpy_array, sampling_rate).
    """
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        sampwidth = wf.getsampwidth()

    # Determina il tipo dati
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.uint8

    audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
    # Normalizza in [-1.0, 1.0]
    max_val = np.iinfo(dtype).max
    audio = audio / max_val

    return audio, sr

def predict_accent_from_url(url: str):
    """
    1) Download MP4
    2) Estrai audio → WAV
    3) Carica WAV con wave+numpy
    4) Preprocess + inferenza con Wav2Vec2
    5) Prendi softmax top-1
    6) Pulisci temporanei
    """
    mp4_file = wav_file = None
    try:
        mp4_file = download_video(url)
        wav_file = extract_audio(mp4_file)

        audio, sr = load_wav(wav_file)        # numpy array [time]

        inputs = feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]
        probs  = torch.softmax(logits, dim=-1)

        idx        = int(probs.argmax())
        label      = model.config.id2label[idx]
        confidence = float(probs[idx] * 100)  # percentuale

        return label, confidence

    finally:
        for path in (mp4_file, wav_file):
            if path and os.path.exists(path):
                os.remove(path)

