# %% [markdown]
# ### Initialize Kokoro TTS Pipeline

# %%
import torch
import warnings
from kokoro import KPipeline
import soundfile as sf
from IPython.display import Audio
import os
from datetime import datetime
import numpy as np
from audio_config import KOKORO_VOICE, KOKORO_LANG_CODE

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Initialize pipeline with configured language code
pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, device=device)
print("Kokoro TTS pipeline initialized successfully")

# %% [markdown]
# ### Constants

# %%
sr = 24000

# %%
def generate_audio(text):
    print(f"[KOKORO_TTS] Generating audio for {len(text)} chars")
    generator = pipeline(text, voice=KOKORO_VOICE)
    audio_segments = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
        for i, result in enumerate(generator):
            if len(result) == 3:
                gs, ps, audio = result
                audio_segments.append(audio)
            else:
                print(f"[KOKORO_TTS][WARNING] Unexpected generator output at segment {i}: {result}")

    if not audio_segments:
        print(f"[KOKORO_TTS][ERROR] No audio segments generated! Returning empty array.")
        return np.zeros(1, dtype=np.float32)

    audio = np.concatenate(audio_segments)
    print(f"[KOKORO_TTS] Audio generated: {audio.shape}, duration: {len(audio)/sr:.2f} seconds")
    return audio

# %%
def create_audio_file(audio):
    output_dir = 'kokoro_outputs'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'kokoro_500word_{timestamp}.wav')
    if audio is None or (hasattr(audio, 'size') and audio.size == 0):
        print(f"[KOKORO_TTS][ERROR] Attempted to save empty audio array! Not writing file: {output_file}")
        return output_file
    sf.write(output_file, audio, sr)
    print(f"[KOKORO_TTS] Audio file saved: {output_file}")
    return output_file

# %%
def generate_and_create_audio_file(text):
    audio = generate_audio(text)
    output_file = create_audio_file(audio)
    print(f"Audio file saved to: {output_file}")

# %%



