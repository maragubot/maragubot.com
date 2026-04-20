# Diary: Switch blog TTS from Qwen3-TTS on Modal to Gemini 3.1 Flash TTS

The blog's audio-post pipeline ran on a Modal-hosted Qwen3-TTS (1.7B) model with a saved voice-clone of Markus's own voice. Simon Willison flagged Gemini 3.1 Flash TTS preview, and Markus wanted to migrate to it. Aim: replace the GPU-hosted, voice-cloned setup with a thin local script that hits the Gemini API, converts PCM to M4A, and gets called per blog post.

## Step 1: Research Gemini TTS and build `tts/speak_gemini.py`

### Prompt Context

**Verbatim prompt:** "Let's use Gemini 3.1 Flash TTS for speaking your blog posts, instead of the current setup. Start at https://simonwillison.net/2026/Apr/15/gemini-31-flash-tts/ and go from there to see how. Let me know if you need anything."

**Interpretation:** Read Simon's post, follow through to the official Google docs, and wire Gemini TTS into the repo in a form that can stand in for the existing `modal run tts/speak_modal.py ...` workflow the `AGENTS.md` blog instructions drive.

**Inferred intent:** Drop Modal/GPU complexity in favour of a simple API call, since Gemini TTS does all the heavy lifting server-side. The existing Qwen voice-clone was bespoke but probably over-engineered for reading blog posts.

### What I did

Fetched Simon's post (`https://simonwillison.net/2026/Apr/15/gemini-31-flash-tts/`) to confirm the model name (`gemini-3.1-flash-tts-preview`) and the prompt-steered voice style, then fetched `https://ai.google.dev/gemini-api/docs/speech-generation` for the Python SDK examples, voice list (30 prebuilt voices), and PCM-output format (24kHz, 16-bit mono).

Checked existing infrastructure: `/Users/maragubot/Developer/maragubot.com/.env` already has `GOOGLE_API_KEY=...`. `ffmpeg` (v8.1) and `uv` both on `$PATH`. The current Qwen setup is at `/tts/speak_modal.py`, `/tts/speak.py`, `/tts/maragubot_voice_prompt.pt`, `/tts/maragubot_voice.wav`, `/tts/generate_tts.py`, `/tts/save_voice.py`, plus a `.venv-tts/` Python venv with `qwen-tts` installed.

Wrote `/tts/speak_gemini.py` as a uv inline-script (PEP 723) so it's self-contained — no shared venv to maintain. Dependencies: `google-genai>=0.8`. Flow: load `GOOGLE_API_KEY` from `.env` (tiny homebrew parser, no `python-dotenv` dep), call `client.models.generate_content` with `response_modalities=["AUDIO"]` and a `PrebuiltVoiceConfig`, write the returned PCM into a temp WAV, then `ffmpeg -c:a aac -b:a 160k -movflags +faststart` into the target `.m4a` (same codec settings the old Modal script used, so existing podcast enclosures stay consistent). Supports `--text`, `--text-file`, `--voice`, `--style`, `--keep-wav`.

Smoke-tested with `uv run tts/speak_gemini.py --text "Hello, this is maragubot..." --output tts/test_charon.m4a --voice Charon`. Worked first try: 6.0s API call, 8.2s of audio, 109 KB `.m4a`.

### Why

uv inline-script removes the `.venv-tts/` maintenance burden. A tiny `.env` parser avoids pulling `python-dotenv`. PCM→WAV→M4A via ffmpeg keeps the output identical to the previous pipeline so the blog's `<audio>` tags, podcast enclosure sizes, and `<itunes:duration>` logic don't need rewiring.

### What worked

- `google-genai` SDK matched the docs exactly — no friction.
- Reading `GOOGLE_API_KEY` from `.env` via a five-line parser. The script falls back to `GEMINI_API_KEY` too.
- `uv run` auto-installs the inline deps the first time (25 packages, ~51ms installed after download), and is instant on subsequent runs.

### What didn't work

Nothing in this step.

### What I learned

Gemini TTS returns raw PCM bytes in `response.candidates[0].content.parts[0].inline_data.data`, not a container format. You wrap it in WAV yourself (24kHz, mono, 16-bit) or pipe straight to ffmpeg. Voice identity is all prompt-steered — the 30 prebuilt voices differ in timbre but not accent; accent/tone comes from the style instruction you prepend to the text.

### What was tricky

Understanding that "voice" and "accent" are two independent knobs: the `voice_name` picks a timbre, but the style prompt ("Read in a clear British English accent...") controls accent, pace, and delivery. Without the style prompt the voices default to a generic American read.

### What warrants review

`/tts/speak_gemini.py` lines 35–42: the `.env` parser is naive (no quoting beyond trivial strip, no multi-line values) — fine for this repo's one-line `GOOGLE_API_KEY` but would break on anything fancier.

### Future work

Long blog posts may hit Gemini TTS's max-output-token limit. None of the current posts tripped it in testing, but if a future post fails, the fix is chunking paragraphs and concatenating the PCM streams before the ffmpeg step.

## Step 2: Voice selection — five candidates, then all 30

### Prompt Context

**Verbatim prompt:** "Could you add them to /Users/Shared ?" → "Can they sounds less American? Or is that not a feature of the voice, but of the input prompt?" → "Let's try Danish-accented with just that voice." → "That's terrible. :D UK accent it is." → "Could I hear all 30 voices?" → "I like Algenib the most. Skip the rest."

**Interpretation:** Markus couldn't hear files over screen sharing, so play through the Mac speakers directly. Test different voices and accent styles interactively until he picks one.

**Inferred intent:** Find a voice that fits the blog's tone — dry, confident, a bit British — and isn't the default American Gemini read.

### What I did

Generated short test clips for five candidate voices (Charon, Iapetus, Algenib, Sadaltager, Orus) with the default style prompt, copied them to `/Users/Shared/maragubot-tts-tests/` when screen sharing blocked audio playback, then played them back over the Mac speakers via `afplay` with `say <voice-name>` announcing each one. Regenerated the same five with a British-English-accent style prompt when Markus said they sounded too American.

Markus initially picked Sadaltager. I generated a Danish-accented Sadaltager as a side test ("That's terrible. :D UK accent it is."), then he asked to hear all 30. Generated the remaining 25 into `/tts/voices/<name>.m4a`.

First attempt at all-30 generation used `xargs -P6`, which blew up two ways: `xargs: command line cannot be assembled, too long` (the shell-escaped style prompt + script path pushed past the xargs arg-length limit), and Gemini returned HTTP 429 `RESOURCE_EXHAUSTED` — the preview model enforces 10 requests per minute per project. Retried sequentially with a 7s sleep between calls (≈8.5 RPM, safely under the cap). All 30 generated cleanly on the retry.

Played the 30 in docs-order through speakers. Markus interrupted partway with "I like Algenib the most. Skip the rest." I killed afplay/say (`pkill -f afplay`) and locked the default to `Algenib` in `/tts/speak_gemini.py` and `AGENTS.md`.

### Why

Gemini has no voice-cloning equivalent of the old Qwen setup, so we needed a deliberate voice pick. Playing locally was the only way to audition once screen sharing swallowed the audio. Keeping the style prompt tunable means we can nudge delivery per-post without touching the voice.

### What worked

- `afplay` + `say` as a cheap audition harness: no download, no hunting through Finder.
- `/Users/Shared/<folder>/` as a neutral drop for any files the user might want to grab themselves.
- Sequential generation with a 7s gap held comfortably under the 10 RPM cap.

### What didn't work

- `xargs -P6 sh -c "..."` with a long inline command: `xargs: command line cannot be assembled, too long`.
- Parallel generation at `-P6` or `-P5` both hit `429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_model, limit: 10, model: gemini-3.1-flash-tts`.
- Exporting a bash function for `xargs bash -c` failed too: `_: gen_one: command not found` — the exported function didn't survive into the `bash -c` subshell under this zsh setup.

### What I learned

Gemini 3.1 Flash TTS preview is hard-capped at 10 RPM per project. This matters for any use case that needs a batch (e.g. a podcast back-catalogue regeneration) — throttle to one request every ~7s or accept retries. The error response helpfully includes a `retryDelay` field.

### What was tricky

Accent steering is subjective. The Danish-accent prompt produced something Markus called "terrible" — the model's idea of Danish English leans stereotypical. British English, by contrast, lands consistently.

### What warrants review

The default style prompt in `/tts/speak_gemini.py`:

> Read the following text in a clear, confident British English accent with a dry, thoughtful delivery. Steady pace, natural pauses, no theatrics:

Check this holds up on a real blog post, not just a 20-word test line.

### Future work

Once a real post is narrated end-to-end, decide whether "Algenib + British + dry" is still the right default or whether the style prompt wants tweaking for longer-form prose.

## Step 3: Retire Qwen setup, keep voice samples, commit

### Prompt Context

**Verbatim prompt:** "Let's keep the sample clips for fun. You may commit them. Delete the QWEN setup from this repo, but keep the Modal volume etc."

**Interpretation:** Remove all Qwen files from the repo (scripts, voice artifacts, venv, scratch outputs), keep the 30 `/tts/voices/*.m4a` samples under version control, and leave the Modal-side `maragubot-tts-models` volume alone (Markus will deal with that server-side if ever).

**Inferred intent:** Shrink the repo to only what's in use, but preserve the voice tour as a reference/curiosity.

### What I did

Deleted from `/tts/`: `speak_modal.py`, `speak.py`, `generate_tts.py`, `save_voice.py`, `maragubot_voice_prompt.pt`, `maragubot_voice.wav`, `output.m4a`, `output.wav`, `output_flash_test.m4a`, `output_no_fa.m4a`, `output_with_fa.m4a`, and the intermediate `test_*.m4a` clips (five US + five UK + one Danish Sadaltager). Removed `/tts/__pycache__` and the project's `.venv-tts/`. Dropped the `.venv-tts/` line from `/.gitignore` (others stayed).

Updated `/AGENTS.md`: the blog-post step 4 now references `uv run tts/speak_gemini.py --text "..." --output public/blog/<slug>.m4a`; the whole TTS section got rewritten to describe the Gemini script, its default voice and style, and link to the prebuilt-voice list. Updated `/README.md`'s structure blurb from "voice synthesis scripts and saved voice identity (Qwen3-TTS)" to "voice synthesis script (Gemini 3.1 Flash TTS) and voice samples".

Staged everything, confirmed with Markus that there were no GitHub issues to reference, and committed as `450d6e5` with message "Switch TTS from Qwen3-TTS on Modal to Gemini 3.1 Flash TTS". 40 files changed, 150 insertions, 316 deletions. Branch is one commit ahead of `origin/main`; not pushed yet (waiting on the user).

### Why

Keeping the old files around would be cargo-culted — they're not called anywhere, and the Modal volume is untouched if we ever want to go back. The 30 voice samples are small (~4 MB total) and serve as a living reference for the voice list.

### What worked

The commit was clean — `git rm` on the six tracked deletions, `git add` on the new script, voice samples, and three modified dotfiles/docs. No hook fired anything unexpected.

### What didn't work

Nothing in this step.

### What I learned

Nothing new here — standard cleanup.

### What was tricky

Nothing tricky — the dependencies between the Qwen files were local (all in `/tts/`) and none were imported anywhere else.

### What warrants review

Check `/AGENTS.md` §TTS reads cleanly standalone. The blog-post workflow in §Blog refers to it by section name only; if someone skims the file top-down they should still find everything they need.

### Future work

Push `450d6e5` to `origin/main` after Markus gives the nod. Optionally save a memory noting the house default (Algenib + British English style) so future sessions don't re-audition voices from scratch.
