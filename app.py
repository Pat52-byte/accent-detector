import streamlit as st
from models import predict_accent
from utils import download_and_extract_audio

st.set_page_config(page_title="English Accent Detector", layout="centered")
st.title("🎙️ English Accent Classifier")
st.write("Upload a `.wav` file or paste a public video URL to detect the English accent.")

input_method = st.radio("Select input method:", ["Upload .wav file", "Paste video URL (MP4, Loom, etc.)"])

if input_method == "Upload .wav file":
    uploaded_file = st.file_uploader("📤 Upload a .wav audio file", type=["wav"])
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Analyzing the accent..."):
            label, confidence = predict_accent("temp.wav")
        st.success(f"🗣️ Detected Accent: **{label}**\n✅ Confidence: **{confidence:.2f}%**")

else:
    video_url = st.text_input("🌐 Paste a public video URL (e.g., Loom, MP4 link)")
    if st.button("Analyze Video") and video_url:
        with st.spinner("Downloading and analyzing audio..."):
            audio_path = download_and_extract_audio(video_url)
            if audio_path:
                label, confidence = predict_accent(audio_path)
                st.success(f"🗣️ Detected Accent: **{label}**\n✅ Confidence: **{confidence:.2f}%**")
            else:
                st.error("❌ Failed to download or extract audio from the video.")

