import streamlit as st
from extract_audio import process_video_from_url
from models import predict_accent

st.set_page_config(page_title="English Accent Classifier", layout="centered")

st.title("🎙️ English Accent Classifier from Video URL")
st.write("Paste a public video URL (e.g., Loom, MP4) to analyze the English accent.")

video_url = st.text_input("🔗 Video URL (must be direct and public):")

if st.button("Analyze Accent"):
    if not video_url:
        st.warning("Please enter a valid video URL.")
    else:
        with st.spinner("Downloading and extracting audio..."):
            audio_path, error = process_video_from_url(video_url)

        if error:
            st.error(f"Error: {error}")
        else:
            with st.spinner("Analyzing accent..."):
                label, confidence, explanation = predict_accent(audio_path)

            st.success("Accent analysis completed.")
            st.markdown(f"**🗣️ Detected Accent:** {label}")
            st.markdown(f"**✅ Confidence:** {confidence:.2f}%")
            st.markdown(f"**ℹ️ Summary:** {explanation}")



