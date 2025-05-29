import torchaudio
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import joblib

# load processor and base model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# load a dummy classifier
classifier = joblib.load("accent_classifier.pkl")  # puoi addestrarlo localmente con scikit-learn
encoder = joblib.load("label_encoder.pkl")

def predict_accent(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = feature_extractor(**inputs).last_hidden_state.mean(dim=1)
    pred = classifier.predict(embedding.numpy())[0]
    proba = classifier.predict_proba(embedding.numpy()).max() * 100
    return encoder.inverse_transform([pred])[0], round(proba, 2)
