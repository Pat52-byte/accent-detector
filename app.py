import streamlit as st
from models import predict_accent
import os

st.title("🎙️ English Accent Detector")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    file_path = os.path.join("temp.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    accent, confidence = predict_accent(file_path)
    st.success(f"Accent: {accent} — Confidence: {confidence:.2%}")

