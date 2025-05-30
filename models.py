from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import moviepy.editor as mp
import requests
import os
import uuid

# Inizializza il classificatore
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="pretrained_models/lang-id-commonlanguage_ecapa",
    run_opts={"device": "cpu"},
    use_auth_token=False
)

def download_video(url):
    temp_video_path = f"temp_{uuid.uuid4()}.mp4"
    response = requests.get(url, stream=True)
    with open(temp_video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_video_path

def extract_audio(video_path):
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, fps=16000, codec='pcm_s16le')  # 16 kHz, PCM WAV
    return temp_audio_path

def predict_accent_from_url(url):
    try:
        video_path = download_video(url)
        audio_path = extract_audio(video_path)
        label, score = classifier.classify_file(audio_path)
        return label, float(score)
    finally:
        # Pulizia file temporanei
        for f in [video_path, audio_path]:
            if os.path.exists(f):
                os.remove(f)

