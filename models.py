import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, AutoProcessor

MODEL_ID = "dima806/english_accents_classification"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

def predict_accent(audio_path):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)

    inputs = processor(speech[0], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_id].item()

    label = model.config.id2label[predicted_id]
    return label, confidence


