from IPython import display as ipd
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
import torchaudio
import os

# Define the output directory
output_dir = r'C:\Users\thanh\Work\Projects\AudioCraft\output'

model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
model.set_generation_params(duration=30)

res = model.generate([
    'crazy EDM, heavy bang', 
    'classic reggae track with an electronic guitar solo',
    'lofi slow bpm electro chill with organic samples',
    'rock with saturated guitars, a heavy bass line and crazy drum break and fills.',
    'earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves',
], progress=True)

# Save the audio files
for i, audio in enumerate(res):
    audio_cpu = audio.cpu()
    file_path = os.path.join(output_dir, f'audio_{i}.wav')
    torchaudio.save(file_path, audio_cpu, sample_rate=32000)

# Display the saved audio files
for i in range(len(res)):
    file_path = os.path.join(output_dir, f'audio_{i}.wav')
    audio, sample_rate = torchaudio.load(file_path)
    display_audio(audio, sample_rate=sample_rate)