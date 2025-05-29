import streamlit as st
from models import predict_accent
from utils import download_and_extract_audio  # crea questa funzione in utils.py

st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎙️ English Accent Classifier")
st.write("Carica un file `.wav` o incolla il link a un video per scoprire l'accento inglese.")

option = st.radio("📥 Scegli input:", ["Carica file .wav", "Incolla link video (MP4, Loom, ecc.)"])

if option == "Carica file .wav":
    uploaded_file = st.file_uploader("📤 Carica un file audio .wav", type=["wav"])
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Analisi in corso..."):
            label, confidence = predict_accent("temp.wav")
        st.success(f"🗣️ Accent: **{label}**\n✅ Confidence: **{confidence:.2f}%**")

else:
    video_url = st.text_input("🌐 Incolla il link del video pubblico (MP4, Loom, ecc.)")
    if st.button("Analizza video") and video_url:
        with st.spinner("Scarico audio e analizzo..."):
            audio_path = download_and_extract_audio(video_url)
            if audio_path:
                label, confidence = predict_accent(audio_path)
                st.success(f"🗣️ Accent: **{label}**\n✅ Confidence: **{confidence:.2f}%**")
            else:
                st.error("❌ Impossibile scaricare o estrarre audio dal video.")

