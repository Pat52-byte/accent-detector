
Accent Detector
A simple tool to evaluate English accents from a public video.

Description
Accent Detector lets you:

Accept a public video URL (Loom, direct .mp4 link, etc.).

Extract the first 10 seconds of audio (mono 16 kHz WAV) using FFmpeg.

Analyze the audio with a fine-tuned Wav2Vec2 model to classify English accents (British, American, Australian, etc.).

Return:

Accent: top-1 label (e.g. british, american, australian)

Confidence: softmax probability converted to percent (0–100%)

This tool is designed to support hiring workflows by quickly identifying the speaker’s English accent.

Project Structure
accent-detector/
├── app.py # Streamlit front-end
├── models.py # Core logic: download, extract audio, inference
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Requirements
Python ≥ 3.8

Internet connection (to download the model and video)

No manual FFmpeg install required (bundled via imageio-ffmpeg)

Installation
Clone this repository:
git clone https://github.com/Pat52-byte/accent-detector.git
cd accent-detector

(Optional) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

Install dependencies:
pip install -r requirements.txt

Usage
Streamlit UI
Launch the app:
streamlit run app.py

In the browser, paste a public .mp4 URL (e.g. Loom) into the input box and click Analyze Accent.

Wait a few seconds—then you’ll see:

Accent (e.g. Australian)

Confidence (e.g. 47.2%)

Python Function (CLI or Script)
You can also import and call the core function in your own script:

from models import predict_accent_from_url
label, confidence = predict_accent_from_url("https://.../video.mp4")
print(f"Accent: {label}, Confidence: {confidence:.1f}%")

Key Files
models.py

download_video(url): downloads the MP4 to a temp file

extract_audio(path): extracts first 10 s of WAV at 16 kHz via FFmpeg

predict_accent_from_url(url): orchestrates download, extraction, inference → returns (label, confidence)

app.py
Streamlit interface that takes a video URL, calls predict_accent_from_url, and displays the result.

requirements.txt

streamlit
transformers
torch
torchaudio
requests
imageio-ffmpeg

Possible Improvements
Support authenticated links (private Loom).

Add an ASR step to measure fluency and pronunciation quality.

Expand the model to more English variants (e.g. Indian, Irish, South African).

Provide a CLI entrypoint (main.py) and automated tests.

License
This project is released under the MIT License.