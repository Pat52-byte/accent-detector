import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Carica il modello di classificazione degli accenti
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    savedir="tmpmodel",
    run_opts={"use_symlink": False}
)


def predict_accent(audio_file):
    # Carica il file audio
    signal, fs = torchaudio.load(audio_file)
    # Esegui la classificazione
    prediction = classifier.classify_file(audio_file)
    label = prediction[3][0]  # etichetta dell'accento
    score = float(prediction[1][0].item()) * 100  # percentuale
    explanation = f"The speaker's accent was classified as {label} with a confidence of {score:.2f}%."
    return label, score, explanation
