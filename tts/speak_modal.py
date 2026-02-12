"""Generate speech in maragubot's voice on Modal (GPU cloud)."""

import modal
import time

app = modal.App("maragubot-tts")

model_volume = modal.Volume.from_name("maragubot-tts-models", create_if_missing=True)
MODEL_CACHE = "/root/models"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("qwen-tts", "soundfile", "numpy")
    .add_local_file("tts/maragubot_voice_prompt.pt", "/root/voice_prompt.pt")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={MODEL_CACHE: model_volume},
)
def generate_speech(text: str) -> tuple[bytes, dict]:
    import os
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

    t0 = time.time()

    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = os.path.join(MODEL_CACHE, "Qwen3-TTS-12Hz-1.7B-Base")

    if not os.path.exists(model_path):
        print(f"Downloading {model_name} to volume...")
        snapshot_download(model_name, local_dir=model_path)
        model_volume.commit()
    else:
        print("Model found in volume cache.")

    print("Loading Base model...")
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    t_model = time.time()
    print(f"Model loaded in {t_model - t0:.1f}s")

    print("Loading voice prompt...")
    payload = torch.load("/root/voice_prompt.pt", map_location="cpu", weights_only=True)
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

    t_voice = time.time()
    print(f"Voice prompt loaded in {t_voice - t_model:.1f}s")

    print(f"Generating speech for: {text[:80]}...")
    t_infer_start = time.time()
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=items,
    )
    t_infer = time.time()
    audio_duration = len(wavs[0]) / sr
    print(f"Inference: {t_infer - t_infer_start:.1f}s for {audio_duration:.1f}s of audio "
          f"(RTF: {(t_infer - t_infer_start) / audio_duration:.2f}x)")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = f"{tmp}/output.wav"
        m4a_path = f"{tmp}/output.m4a"

        sf.write(wav_path, wavs[0], sr)

        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-c:a", "aac", "-b:a", "160k",
            "-vn", "-sn", "-dn",
            "-movflags", "+faststart",
            m4a_path,
        ], check=True, capture_output=True)

        with open(m4a_path, "rb") as f:
            audio_bytes = f.read()

    t_total = time.time()
    stats = {
        "model_load_s": round(t_model - t0, 1),
        "inference_s": round(t_infer - t_infer_start, 1),
        "audio_duration_s": round(audio_duration, 1),
        "rtf": round((t_infer - t_infer_start) / audio_duration, 2),
        "total_s": round(t_total - t0, 1),
        "file_size_bytes": len(audio_bytes),
    }
    print(f"\nStats: {stats}")
    return audio_bytes, stats


@app.local_entrypoint()
def main(text: str, output: str = "tts/output.m4a"):
    import pathlib

    print(f"Text: {text}")
    print("Running on Modal...")
    audio_bytes, stats = generate_speech.remote(text)

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio_bytes)
    print(f"\nSaved to {out} ({len(audio_bytes)} bytes)")
    print(f"Audio duration: {stats['audio_duration_s']}s")
    print(f"Inference time: {stats['inference_s']}s")
    print(f"Real-time factor: {stats['rtf']}x")
    print(f"Total (including model load): {stats['total_s']}s")
