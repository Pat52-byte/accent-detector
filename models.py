import os
import uuid
import requests
import subprocess

import torch
import torchaudio
import imageio_ffmpeg
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# 1) Path al binario FFmpeg incluso in imageio-ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Caricamento una tantum del feature extractor e del modello
MODEL_ID = "dima806/english_accents_classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

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
    Estrae un WAV mono 16 kHz usando FFmpeg in subprocess.
    """
    wav_path = f"tmp_{uuid.uuid4()}.wav"
    cmd = [
        FFMPEG_EXE,
        "-y",               # sovrascrivi se esiste
        "-i", video_path,   # input file
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def predict_accent_from_url(url: str):
    """
    1) Download del MP4
    2) Estrazione audio → WAV
    3) Caricamento WAV → Tensor via torchaudio
    4) Preprocessing + inferenza con Wav2Vec2ForSequenceClassification
    5) Softmax → top-1 label + confidence
    6) Pulizia dei file temporanei
    """
    mp4_file = wav_file = None
    try:
        # Passi 1 & 2
        mp4_file = download_video(url)
        wav_file = extract_audio(mp4_file)

        # Passo 3: load audio
        waveform, sr = torchaudio.load(wav_file)  # Tensor [1, T]
        waveform = waveform.squeeze(0)            # Tensor [T]

        # Passo 4: preprocessing
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        # Passo 5: inferenza
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]                # Tensor [num_labels]
        probs  = torch.softmax(logits, dim=-1)    # probabilità

        # Seleziona etichetta e confidence
        idx        = int(probs.argmax())
        label      = model.config.id2label[idx]
        confidence = float(probs[idx] * 100)     # percentuale 0–100%

        return label, confidence

    finally:
        # Passo 6: cleanup
        for path in (mp4_file, wav_file):
            if path and os.path.exists(path):
                os.remove(path)
