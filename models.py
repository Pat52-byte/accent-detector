import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

MODEL_ID = "dima806/english_accents_classification"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

def predict_accent(file_path):
    speech, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)

    input_values = processor(speech[0], sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        score = torch.nn.functional.softmax(logits, dim=-1).squeeze()[predicted_id].item()
    
    predicted_label = model.config.id2label[predicted_id]
    return predicted_label, score

