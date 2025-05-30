import os, uuid, requests
from moviepy.editor import VideoFileClip
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

# 1) Download & prepare the HuggingFace accent-ID model
MODEL_ID = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model     = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
accent_pipe = pipeline(
    "audio-classification",
    model=model,
    feature_extractor=extractor,
    top_k=1,
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
        clip.audio.write_audiofile(wav, fps=16000, codec="pcm_s16le")
    return wav

def predict_accent_from_url(url: str):
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)
        result = accent_pipe(ap)[0]
        label = result["label"]
        score = float(result["score"])
        return label, score
    finally:
        for f in (vp, ap):
            if f and os.path.exists(f):
                os.remove(f)


