import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os

# MODELLO FINE-TUNED (es. dialetti o accenti)
MODEL_ID = "sauravjoshi/Accent-Classifier-Wav2Vec2"


# Carica modello e processore
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

# Percorso file
files = ["audio_uk_1.wav.wav", "audio_us_1.wav.wav"]

# Analisi
for file in files:
    print(f"\n🎧 Analizzando {file}...")

    # Carica audio
    speech, sr = torchaudio.load(file)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)
    
    input_values = processor(speech[0], sampling_rate=16000, return_tensors="pt").input_values

    # Predizione
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        score = torch.nn.functional.softmax(logits, dim=-1).squeeze()[predicted_id].item()

    # Mappatura classi
    labels = model.config.id2label
    predicted_label = labels[predicted_id]

    print(f"🗣️ Accent: {predicted_label} | Confidence: {score:.2%}")
