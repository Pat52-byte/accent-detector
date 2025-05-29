import os
import tempfile
import requests
from moviepy.editor import VideoFileClip

def process_video_from_url(url):
    try:
        # Scarica il file video temporaneamente
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    temp_video.write(chunk)
            video_path = temp_video.name
        
        # Estrai audio in WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        return audio_path
    
    except Exception as e:
        print("Errore durante il download o l'estrazione:", e)
        return None
