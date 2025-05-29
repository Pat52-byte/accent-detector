iimport streamlit as st
from models import predict_accent
from extract_audio import process_video_from_url

st.set_page_config(page_title="English Accent Classifier", layout="centered")

st.title("🎙️ English Accent Classifier")
st.write("Upload a `.wav` file or paste a public video URL to detect the English accent of the speaker.")

option = st.radio("Choose input type:", ("Upload Audio File (.wav)", "Paste Video URL (e.g. Loom or MP4 link)"))

if option == "Upload Audio File (.wav)":
    uploaded_file = st.file_uploader("📤 Upload a `.wav` audio file", type=["wav"])
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Analyzing the accent..."):
            label, confidence = predict_accent("temp.wav")
        st.success(f"🗣️ Detected Accent: **{label}** \n✅ Confidence: **{confidence:.2f}%**")

elif option == "Paste Video URL (e.g. Loom or MP4 link)":
    video_url = st.text_input("🎬 Paste a public video URL:")
    if video_url:
        with st.spinner("Downloading and extracting audio from the video..."):
            audio_path = process_video_from_url(video_url)
        if audio_path:
            with st.spinner("Analyzing the accent..."):
                label, confidence = predict_accent(audio_path)
            st.success(f"🗣️ Detected Accent: **{label}** \n✅ Confidence: **{confidence:.2f}%**")
        else:
            st.error("❌ Unable to process the video. Please check the URL and try again.")



