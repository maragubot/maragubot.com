"""Generate speech in maragubot's saved voice."""

import sys
from dataclasses import asdict

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_PROMPT = "maragubot_voice_prompt.pt"

if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I have nothing to say, which is unusual for me."
output_path = "output.wav"

print(f"Loading Base model...")
tts = Qwen3TTSModel.from_pretrained(
    BASE_MODEL,
    device_map=device,
    dtype=dtype,
    attn_implementation="eager",
)

print(f"Loading voice prompt from {VOICE_PROMPT}...")
payload = torch.load(VOICE_PROMPT, map_location="cpu", weights_only=True)
items = []
for d in payload["items"]:
    ref_code = d.get("ref_code")
    if ref_code is not None and not torch.is_tensor(ref_code):
        ref_code = torch.tensor(ref_code)
    ref_spk = d["ref_spk_embedding"]
    if not torch.is_tensor(ref_spk):
        ref_spk = torch.tensor(ref_spk)
    items.append(VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk,
        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
        icl_mode=bool(d.get("icl_mode", True)),
        ref_text=d.get("ref_text"),
    ))

print(f"Saying: {text}")
wavs, sr = tts.generate_voice_clone(
    text=text,
    language="English",
    voice_clone_prompt=items,
)

sf.write(output_path, wavs[0], sr)
print(f"Saved to {output_path} (sample rate: {sr})")
