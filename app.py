import streamlit as st
from models import predict_accent
import os

st.title("🎤 Accent Detector (English Only)")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    audio_path = os.path.join("temp.wav")
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(audio_path, format="audio/wav")

    with st.spinner("Analyzing..."):
        accent, confidence = predict_accent(audio_path)
        st.success(f"🗣️ Accent: **{accent}**\n\n🔒 Confidence: **{confidence:.2%}**")
