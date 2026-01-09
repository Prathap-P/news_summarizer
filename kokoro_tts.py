# %% [markdown]
# ### Initialize Kokoro TTS Pipeline

# %%
import torch
from kokoro import KPipeline
import soundfile as sf
from IPython.display import Audio
import os
from datetime import datetime
import numpy as np

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Initialize pipeline with American English
pipeline = KPipeline(lang_code='a', device=device)
print("Kokoro TTS pipeline initialized successfully")

# %% [markdown]
# ### Constants

# %%
sr = 24000

# %%
def generate_audio(text):
    generator = pipeline(text, voice='af_heart')
    audio_segments = []

    for i, (gs, ps, audio) in enumerate(generator):
        # print(f"Segment {i}: {len(audio)} samples")
        audio_segments.append(audio)

    # Concatenate all segments
    audio = np.concatenate(audio_segments)

    # print(f"Total duration: {len(audio) / sr:.2f} seconds")
    # print(f"Audio shape: {audio.shape}")
    
    return audio

# %%
def create_audio_file(audio):
    output_dir = 'kokoro_outputs'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'kokoro_500word_{timestamp}.wav')
    # Save audio
    sf.write(output_file, audio, sr)
    return output_file

# %%
def generate_and_create_audio_file(text):
    audio = generate_audio(text)
    output_file = create_audio_file(audio)
    print(f"Audio file saved to: {output_file}")

# %%



