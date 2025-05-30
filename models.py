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

# 1) Point to the bundled ffmpeg binary
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Load the proper feature extractor & model
MODEL_ID = "dima806/english_accents_classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def download_video(url: str) -> str:
    """Download MP4 to a temp file."""
    path = f"tmp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path: str) -> str:
    """
    Use ffmpeg subprocess to extract a mono 16 kHz WAV.
    """
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

def predict_accent_from_url(url: str):
    """
    1) Download video → MP4
    2) Extract audio → WAV
    3) Load WAV with torchaudio → Tensor
    4) Preprocess + infer with Wav2Vec2ForSequenceClassification
    5) Softmax → top-1 (label, score)
    6) Cleanup temp files
    """
    mp4_file = wav_file = None
    try:
        # 1 & 2)
        mp4_file = download_video(url)
        wav_file = extract_audio(mp4_file)

        # 3) load audio
        waveform, sr = torchaudio.load(wav_file)  # → Tensor [1, time]
        waveform = waveform.squeeze(0)            # → Tensor [time]

        # 4) preprocess
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        # 5) inference
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]                # → Tensor [num_labels]
        probs  = torch.softmax(logits, dim=-1)

        # 6) pick top-1
        idx   = int(probs.argmax())
        label = model.config.id2label[idx]
        score = float(probs[idx])

        return label, score

    finally:
        # cleanup
        for path in (mp4_file, wav_file):
            if path and os.path.exists(path):
                os.remove(path)
