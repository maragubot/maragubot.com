#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=0.8",
# ]
# ///
"""Generate speech from text using Gemini 3.1 Flash TTS.

Usage:
    uv run tts/speak_gemini.py --text "Hello" --output out.m4a
    uv run tts/speak_gemini.py --text-file post.txt --output post.m4a --voice Iapetus
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import wave

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_VOICE = "Algenib"
DEFAULT_STYLE = (
    "Read the following text in a clear, confident British English accent with a "
    "dry, thoughtful delivery. Steady pace, natural pauses, no theatrics:"
)


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_wav(path: pathlib.Path, pcm: bytes, rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def wav_to_m4a(wav_path: pathlib.Path, m4a_path: pathlib.Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-c:a", "aac", "-b:a", "160k",
            "-vn", "-sn", "-dn",
            "-movflags", "+faststart",
            str(m4a_path),
        ],
        check=True,
        capture_output=True,
    )


def generate(text: str, voice: str, style: str) -> bytes:
    client = genai.Client()
    prompt = f"{style}\n\n{text}" if style else text
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def main() -> int:
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to speak.")
    src.add_argument("--text-file", type=pathlib.Path, help="Path to a UTF-8 text file.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output .m4a path.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Prebuilt voice (default: {DEFAULT_VOICE}).")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="Optional style prompt prepended to the text.")
    parser.add_argument("--keep-wav", action="store_true", help="Also keep the intermediate WAV file next to the output.")
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("error: GOOGLE_API_KEY (or GEMINI_API_KEY) not set", file=sys.stderr)
        return 1

    text = args.text if args.text else args.text_file.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        print("error: empty text", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Voice: {args.voice}")
    print(f"Chars: {len(text)}")
    print(f"Model: {MODEL}")

    t0 = time.time()
    pcm = generate(text, args.voice, args.style)
    t_api = time.time() - t0
    print(f"API: {t_api:.1f}s, {len(pcm)} PCM bytes")

    audio_duration = len(pcm) / (24000 * 2)
    print(f"Audio duration: {audio_duration:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = pathlib.Path(tmp) / "out.wav"
        write_wav(wav_path, pcm)
        wav_to_m4a(wav_path, args.output)
        if args.keep_wav:
            keep = args.output.with_suffix(".wav")
            keep.write_bytes(wav_path.read_bytes())
            print(f"WAV: {keep}")

    size = args.output.stat().st_size
    total = time.time() - t0
    print(f"Saved: {args.output} ({size} bytes)")
    print(f"Total: {total:.1f}s (RTF {total / max(audio_duration, 0.01):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
