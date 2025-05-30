import os, uuid, requests, shutil, math
import torchaudio, moviepy.editor as mp
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# 1) Accent model, copy-only strategy
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    savedir="pretrained_models/accent-id-commonaccent_xlsr-en-english",
    local_strategy=LocalStrategy.COPY,
    run_opts={"device": "cpu"},
    use_auth_token=False
)

def download_video(url):
    path = f"temp_{uuid.uuid4()}.mp4"
    resp = requests.get(url, stream=True); resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_audio(video_path):
    wav = f"temp_{uuid.uuid4()}.wav"
    with mp.VideoFileClip(video_path) as video:
        video.audio.write_audiofile(wav, fps=16000, codec="pcm_s16le")
    return wav

def predict_accent_from_url(url):
    vp = ap = None
    try:
        vp = download_video(url)
        ap = extract_audio(vp)

        out_prob, log_score, _, label_list = classifier.classify_file(ap)
        # batch size is 1 → take [0]
        label      = label_list[0]
        confidence = math.exp(log_score[0])

        return label, confidence
    finally:
        for f in (vp, ap):
            if f and os.path.exists(f):
                os.remove(f)


