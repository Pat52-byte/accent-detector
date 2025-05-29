import streamlit as st
from utils import download_video, extract_audio, transcribe_and_check_english, get_fluency_score
from models import predict_accent
import pandas as pd
import os

st.title("🎤 AI Accent Detector for English Speakers")

url = st.text_input("Paste a public video URL (MP4)")

if st.button("Analyze"):
    with st.spinner("Downloading video..."):
        video_path = download_video(url)
        audio_path = extract_audio(video_path)

    with st.spinner("Checking if English..."):
        is_english, transcript = transcribe_and_check_english(audio_path)

    if is_english:
        with st.spinner("Analyzing accent..."):
            accent, confidence = predict_accent(audio_path)
            fluency = get_fluency_score(transcript, audio_path)

            st.success("✅ English detected")
            st.write(f"**Predicted Accent:** {accent}")
            st.write(f"**Accent Confidence:** {confidence}%")
            st.write(f"**Fluency Score:** {fluency}/100")
            st.write("**Transcript:**", transcript[:300] + "...")

            df = pd.DataFrame([{
                "Accent": accent,
                "Confidence": confidence,
                "Fluency": fluency,
                "Transcript": transcript[:100] + "..."
            }])

            st.download_button("📥 Download CSV", df.to_csv(index=False), "result.csv", "text/csv")
        os.remove(video_path)
        os.remove(audio_path)
    else:
        st.error("This video is not in English.")
