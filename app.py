import streamlit as st
from models import predict_accent
from extract_audio import process_video_from_url  # <--- salva la tua funzione in extract_audio.py

st.set_page_config(page_title="English Accent Classifier", layout="centered")

st.title("🎙️ English Accent Classifier")
st.write("Paste a public video URL or upload a `.wav` file to detect the speaker's English accent.")

option = st.radio("Choose input method:", ["📤 Upload .wav", "🌐 Paste video URL"])

audio_path = None

if option == "📤 Upload .wav":
    uploaded_file = st.file_uploader("Upload a `.wav` audio file", type=["wav"])
    if uploaded_file:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        audio_path = "temp.wav"

elif option == "🌐 Paste video URL":
    url = st.text_input("Paste a public Loom or MP4 video URL")
    if url:
        with st.spinner("Downloading and extracting audio..."):
            audio_path = process_video_from_url(url)
        if audio_path:
            st.success("✅ Audio extracted successfully!")
        else:
            st.error("❌ Failed to download or extract audio. Please check the URL.")

if audio_path:
    with st.spinner("Analyzing accent..."):
        label, confidence = predict_accent(audio_path)
    st.success(f"🗣️ Accent: **{label}**\n✅ Confidence: **{confidence:.2%}**")


