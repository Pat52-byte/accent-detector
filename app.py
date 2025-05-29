import streamlit as st
from models import predict_accent

st.set_page_config(page_title="English Accent Classifier", layout="centered")

st.title("🎙️ English Accent Classifier")
st.write("Carica un file `.wav` e scopri l'accento inglese.")

uploaded_file = st.file_uploader("📤 Carica un file audio .wav", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Analisi in corso..."):
        label, confidence = predict_accent("temp.wav")
    st.success(f"🗣️ Accent: **{label}**\n✅ Confidence: **{confidence:.2%}**")
