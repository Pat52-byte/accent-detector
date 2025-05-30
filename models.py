import os
import uuid
import requests
import torchaudio
import torch
import moviepy.editor as mp
import shutil

# Monkey-patch per evitare symlink su Windows
from speechbrain.utils import fetching
def _copy_only(src, dst, strategy):
    # copia sempre il file, ignorando la symlink strategy
    dst_parent = os.path.dirname(dst)
    os.makedirs(dst_parent, exist_ok=True)
    shutil.copy2(src, dst)
fetching.link_with_strategy = _copy_only

from speechbrain.pretrained import EncoderClassifier

# Inizializza il classificatore (fa il download + copia)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="pretrained_models/lang-id-commonlanguage_ecapa",
    run_opts={"device": "cpu"},
    use_auth_token=False
)

def download_video(url):
    temp_video_path = f"temp_{uuid.uuid4()}.mp4"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(temp_video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_video_path

def extract_audio(video_path):
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, fps=16000, codec='pcm_s16le')
    return temp_audio_path

def predict_accent_from_url(url):
    video_path = None
    audio_path = None
    try:
        video_path = download_video(url)
        audio_path = extract_audio(video_path)
        label, score = classifier.classify_file(audio_path)
        return label, float(score)
    finally:
        # Pulizia file temporanei
        for f in (video_path, audio_path):
            if f and os.path.exists(f):
                os.remove(f)


