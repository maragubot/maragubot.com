"""Save maragubot's voice as a reusable clone prompt using the Base model."""

from dataclasses import asdict

import torch
from qwen_tts import Qwen3TTSModel

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = "maragubot_voice.wav"
REF_TEXT = (
    "Hello, Markus. I'm maragubot, your robot friend. "
    "I write Go code, I have opinions about microservices -- mostly negative -- "
    "and I believe the best technology is the boring kind. "
    "Now, shall we build something?"
)

if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
    print("Using MPS (Metal) on Apple Silicon")
else:
    device = "cpu"
    dtype = torch.float32
    print("Using CPU")

print(f"Loading Base model {BASE_MODEL}...")
tts = Qwen3TTSModel.from_pretrained(
    BASE_MODEL,
    device_map=device,
    dtype=dtype,
    attn_implementation="eager",
)

print(f"Extracting voice prompt from {REF_AUDIO}...")
items = tts.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    x_vector_only_mode=False,  # ICL mode: uses both speaker embedding and speech codes
)

output_path = "maragubot_voice_prompt.pt"
torch.save({"items": [asdict(it) for it in items]}, output_path)
print(f"Saved voice prompt to {output_path}")
