from moviepy.editor import VideoFileClip

# Lista dei file MP4 da convertire
files = ["audio_uk_1.wav.mp4", "audio_us_1.wav.mp4"]

for f in files:
    # Pulisce il nome: rimuove solo ".mp4" (non ".wav")
    output = f.replace(".mp4", "")
    clip = VideoFileClip(f)
    clip.audio.write_audiofile(output, fps=16000)
    print(f"✅ Creato: {output}")


