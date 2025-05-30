import os
import uuid
import requests
import subprocess

import torch
import torchaudio
import imageio_ffmpeg
from transformers import AutoProcessor, AutoModelForAudioClassification

# 1) Point to the bundled ffmpeg binary
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# 2) Load the model & processor once at import time
MODEL_ID = "dima806/english_accents_classification"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()

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
    cmd = [
        FFMPEG_EXE,
        "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def predict_accent_from_url(url: str):
    mp4_file = wav_file = None
    try:
        # 1) download & extract
        mp4_file = download_video(url)
        wav_file = extract_audio(mp4_file)

        # 2) load with torchaudio → Tensor [1, time]
        waveform, sr = torchaudio.load(wav_file)
        waveform = waveform.squeeze(0)  # [time]

        # 3) prep inputs for the model
        inputs = processor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        # 4) inference
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # [1, num_labels]
        probs  = torch.softmax(logits, dim=-1)[0]

        # 5) pick top-1
        idx   = int(probs.argmax())
        label = model.config.id2label[idx]
        score = float(probs[idx])

        return label, score

    finally:
        # cleanup
        for fpath in (mp4_file, wav_file):
            if fpath and os.path.exists(fpath):
                os.remove(fpath)

