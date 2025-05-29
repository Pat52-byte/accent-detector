import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Dataset simulato: file audio + etichette di accento
data = [
    ("audio_uk_1.wav", "british"),
    ("audio_us_1.wav", "american"),
    ("audio_uk_2.wav", "british"),
    ("audio_us_2.wav", "american"),
    # Aggiungi i tuoi file audio reali qui
]

X = []
y = []

for file, label in data:
    y.append(label)
    signal, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfccs.T, axis=0)
    X.append(mean_mfcc)

# Addestramento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Valutazione
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Salva il modello
joblib.dump(clf, "accent_classifier.pkl")
print("✅ Modello salvato in accent_classifier.pkl")

