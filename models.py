from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa

MODEL_ID = "sauravjoshi/Accent-Classifier-Wav2Vec2"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

def predict_accent(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_id].item()
    label = model.config.id2label[predicted_id]
    return label, confidence

