import streamlit as st
from models import predict_accent_from_url

st.set_page_config(page_title="Accent Detector", layout="centered")
st.title("🎙️ Accent Detector from Video URL")

st.markdown("Enter a public video link (e.g., Loom or .mp4) containing spoken English.")

url = st.text_input("🔗 Video URL (public Loom or .mp4)")

if st.button("Analyze Accent") and url:
    with st.spinner("⏳ Extracting audio and analyzing..."):
        try:
            label, confidence = predict_accent_from_url(url)
            st.success(f"✅ Detected Accent: **{label}**")
            st.info(f"🔍 Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")




