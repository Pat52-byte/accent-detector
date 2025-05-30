import streamlit as st
from models import predict_accent_from_url

st.set_page_config(page_title="Accent Detector", layout="centered")
st.title("🎙️ Accent Detector from Video URL")

st.markdown("Inserisci un link pubblico a un video (es. Loom o file MP4) con parlato in inglese.")

url = st.text_input("🔗 URL del video (Loom o .mp4 pubblico)")

if st.button("Analizza accento") and url:
    with st.spinner("⏳ Estraendo audio e analizzando..."):
        try:
            label, confidence = predict_accent_from_url(url)
            st.success(f"✅ Accento rilevato: **{label}**")
            st.info(f"🔍 Confidenza: {confidence:.2f}")
        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {e}")



