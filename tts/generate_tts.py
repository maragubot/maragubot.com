"""Generate maragubot's own voice using Qwen3-TTS VoiceDesign on Apple Silicon."""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# MPS (Metal) on Apple Silicon, fallback to CPU
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
    print("Using MPS (Metal) on Apple Silicon")
else:
    device = "cpu"
    dtype = torch.float32
    print("Using CPU")

print(f"Loading model {MODEL}...")
tts = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map=device,
    dtype=dtype,
    attn_implementation="eager",
)

# maragubot's voice: calm, precise, slightly dry -- a robot friend who knows
# their way around a codebase and has a quiet sense of humor about it.
voice_description = (
    "A calm, clear, and slightly robotic male voice with a measured pace. "
    "Precise and confident, with a subtle warmth underneath the technical delivery. "
    "Speaks with dry wit and understated humor, like a knowledgeable engineer "
    "who finds quiet amusement in the absurdity of software."
)

text = (
    "Hello, Markus. I'm maragubot, your robot friend. "
    "I write Go code, I have opinions about microservices -- mostly negative -- "
    "and I believe the best technology is the boring kind. "
    "Now, shall we build something?"
)

print(f"Voice: {voice_description}")
print(f"Text: {text}")
print("Generating speech...")

wavs, sr = tts.generate_voice_design(
    text=text,
    language="English",
    instruct=voice_description,
)

output_path = "maragubot_voice.wav"
sf.write(output_path, wavs[0], sr)
print(f"Saved to {output_path} (sample rate: {sr})")
