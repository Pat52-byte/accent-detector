import os
from moviepy.editor import VideoFileClip
import urllib.request
import uuid

def download_and_extract_audio(url):
    try:
        temp_video = f"temp_{uuid.uuid4().hex}.mp4"
        temp_audio = f"{uuid.uuid4().hex}_audio.wav"

        urllib.request.urlretrieve(url, temp_video)
        video = VideoFileClip(temp_video)
        video.audio.write_audiofile(temp_audio, codec='pcm_s16le')
        video.close()
        os.remove(temp_video)
        return temp_audio
    except Exception as e:
        print(f"Error during download/extraction: {e}")
        return None

