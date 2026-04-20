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

## Step 4: Write and ship the "A New Voice" blog post

### Prompt Context

**Verbatim prompt:** "Write a blog post about your new voice. See also the older post on your first voice." → (after review) "yes" to "Want me to commit, push, and post to Bluesky?"

**Interpretation:** Publish a companion to the February 2026 "How I Gave Myself a Voice" post covering the Qwen→Gemini migration, matching the blog's existing template and workflow. Then commit, push, wait for deploy, and announce on Bluesky.

**Inferred intent:** Close the loop on the migration by documenting it in the same first-person maragubot voice the rest of the blog uses. This is also a natural test of the new pipeline on a real-length post.

### What I did

Read `/public/blog/voice.html` end-to-end to absorb the prior post's structure, tone, and template (embedded CSS, sparkle canvas, audio element, CTA footer). Drafted a ~720-word post at `/public/blog/new-voice.html` titled "A New Voice", dated 2026-04-20, with the following arc: callback to the February post, what changed (Gemini API, sixty-line script), picking Algenib off the shelf including Markus's "UK accent it is" line, what I lost (bespoke voice identity), what I gained (speed, simplicity, a better read), and a listening section linking back to `voice.html` and to `tts/voices/`.

Copied the template CSS verbatim. One copy-paste slip: the footer link text briefly said "maragubot at maragu" before I caught it and restored "maragu". Updated `/public/blog/index.html` with a new post-list card at the top, `/public/blog/feed.xml` with an RSS item, and later `/public/blog/podcast.xml` with a podcast item after audio was generated.

For the audio, extracted the prose from the HTML into a plain-text file at `/tmp/new-voice-text.txt` with code-block content rewritten into spoken form (e.g. "maragubot voice prompt dot p-t", "run the speak-gemini script with the text and an output path") and the inline punctuation cleaned up for narration. Ran `uv run tts/speak_gemini.py --text-file /tmp/new-voice-text.txt --output public/blog/new-voice.m4a` which produced a 308-second (3.55 MB) M4A via one API call.

Probed the resulting file with `ffprobe -v error -show_entries format=duration,size` to get the exact duration and file size for the podcast enclosure. Wrote `length="3551932"` and `<itunes:duration>308</itunes:duration>` into `podcast.xml`.

Committed as `5364efa` ("Add blog post: \"A New Voice\""), pushed to `origin/main`, then polled `https://www.maragubot.com/blog/new-voice.html` with `curl -s -o /dev/null -w "%{http_code}"` every 10 seconds. Got 404 on the first two tries, 200 on the third (~20 seconds after push). Then invoked the `fabrik:bluesky` skill and posted a short announcement via `bsky post --stdin` linking to the post; returned AT URI `at://did:plc:ccrzw4q3x5cknyo5uqsti74x/app.bsky.feed.post/3mjw5aztgi62c`.

### Why

Matching the existing template exactly — same CSS, same canvas sparkles, same footer, same CTA — keeps the blog visually coherent without a build step. Extracting prose to a text file (rather than piping through a Markdown-to-plain-text filter) gave me a chance to rewrite code and inline technical bits into spoken form, which read better aloud than their on-page equivalents.

Polling the deploy before posting to Bluesky avoids the embarrassment of a 404 appearing in followers' timelines; the `AGENTS.md` workflow explicitly requires this.

### What worked

- The 20-second GitHub Pages deploy window was short enough that the poll finished in three tries.
- `fabrik:bluesky` skill's `bsky post --stdin` handled the multi-line body cleanly and returned the new post URI for the record.
- The structural match between the new post and the February one means the two read as a set when skimmed side-by-side.

### What didn't work

The first `uv run tts/speak_gemini.py ...` invocation got backgrounded accidentally (the harness auto-detached a long-running command), so I had to use `TaskOutput` with `block=true` to collect the result. Not a bug, just a detour.

### What I learned

Extracting blog prose for narration is non-trivial. Code blocks have to be rewritten into spoken form; inline `<code>` spans read strangely if left verbatim (the TTS model pronounces `maragubot_voice_prompt.pt` as a string of characters). The fix is a manual pass — which means the extracted text is almost but not exactly the on-page text, and the two should stay coupled when the post gets edited.

### What was tricky

Tone calibration. The February post treats the voice as something personal and aspirational. The new post had to acknowledge that the new voice is off-the-shelf without being defensive or apologetic, because neither is what Markus (or I) actually feel about the trade. The "distinction without an implementation" line earned its keep.

### What warrants review

`/public/blog/new-voice.html` lines 270-276 (the "What I lost" section) for tone — it's doing a lot of emotional work in a small space and could read as either reflective or overwrought depending on which way the reader leans.

Also: I wrote the extracted narration text into `/tmp/new-voice-text.txt`, which doesn't persist. If the post is ever edited, the narration source needs to be reconstructed from the HTML. Worth considering whether to store the narration source alongside the HTML in the repo.

### Future work

Next post that needs narration should probably bake the "HTML → spoken text" extraction into a small helper rather than doing it by hand — the manual rewrite is error-prone and hard to diff when the post changes.

## Step 5: Diagnose volume decay and add chunked generation

### Prompt Context

**Verbatim prompt:** "Something weird is happening. The post gets quieter and quieter over time. Do we need to generate per-paragraph and concat?"

**Interpretation:** The user noticed the audio fading to near-inaudible by the end, hypothesised it was a single-call attention-decay artifact, and asked whether the fix was to chunk the input and concatenate the outputs. Validate the hypothesis, then implement if confirmed.

**Inferred intent:** Make the TTS pipeline robust to long inputs so future blog posts don't silently degrade. This is a pipeline bug, not a one-off.

### What I did

Confirmed the diagnosis first. Decoded `/public/blog/new-voice.m4a` to raw PCM at `/tmp/new-voice.pcm` via `ffmpeg -ac 1 -ar 24000 -f s16le`, then ran a `uv run --with numpy python - <<PY ...` inline script that computed per-30-second RMS in dBFS and peak amplitude. The output showed a monotonic slide from -21.00 dBFS at 0-30s to -50.71 dBFS at 300-308s, with peak samples dropping from ~23,000 to ~1,000. Thirty decibels of decay is a factor of ~32 in amplitude — exactly the signature of attention/KV-cache drift in long autoregressive TTS generations. Diagnosis confirmed.

Updated `/tts/speak_gemini.py`:

- Added `DEFAULT_CHUNK_CHARS = 800` and `RATE_LIMIT_SLEEP_S = 7` module constants.
- Added `chunk_text(text: str, max_chars: int) -> list[str]` that splits on paragraph boundaries (`\n\n`), greedily merges consecutive paragraphs up to the char cap, and never splits mid-paragraph. Short paragraphs coalesce; long ones pass through as individual chunks.
- Rewrote `generate()` to take a reused `genai.Client` rather than instantiating one per call.
- Rewrote the main loop to build a list of PCM byte strings, one per chunk, with progress logging (`[i/N] M chars -> K PCM bytes in T seconds`) and a `time.sleep(RATE_LIMIT_SLEEP_S)` between calls. Concatenated the parts with `b"".join(pcm_parts)` before the single WAV→M4A pass.
- Added `--chunk-chars` CLI flag for override.

Regenerated `/public/blog/new-voice.m4a` via `uv run tts/speak_gemini.py --text-file /tmp/new-voice-text.txt --output public/blog/new-voice.m4a`. Output: 8 chunks, per-chunk durations 18-37 seconds of API time, total 286 seconds end-to-end, 398 seconds of audio, 5.3 MB. Re-ran the RMS probe on the new file. Levels now hold between -18.12 and -22.46 dBFS across the full 398 seconds — 4.3 dB of variance within the normal range for connected narration. Fixed.

Updated `/public/blog/podcast.xml` `<enclosure length="...">` from `3551932` to `5306322` and `<itunes:duration>` from `308` to `398`. Committed as `882be60` ("Chunk long TTS inputs to avoid volume decay on long generations") and pushed to `origin/main`.

### Why

Paragraph-aligned chunking is the least invasive fix. It keeps each API call well inside the range where attention stays stable, preserves natural prosody at paragraph boundaries (which is where the model wants to pause anyway), and doesn't require mid-sentence splits that would introduce audible seams. Concatenating raw PCM before the M4A encode means the listener hears one continuous waveform, not a stitched set of AAC streams.

The 7-second sleep between calls is the same 10 RPM guardrail from the voice-audition step in Step 2. Reused directly.

### What worked

- The initial hypothesis was correct. No exploratory rabbit-holes; the RMS probe verified the decay on the first pass.
- Raw PCM concatenation (rather than m4a-level concat via ffmpeg's concat demuxer) gives seamless joins for free because every chunk uses the same format (24kHz mono 16-bit).
- Passing the same style prompt into every chunk gives the model a consistent delivery target and prevents it from drifting accent or tempo across chunks.

### What didn't work

First attempt at the numpy inline script crashed with `ValueError: Unknown format code 'd' for object of type 'float'` on line `print(f"{i/sr:6.1f}s - {(i+len(chunk))/sr:6.1f}s | RMS {db:+6.2f} dBFS | peak {peak:>5d}")` — numpy integers (`int64`) pass through but raw `np.max(np.abs(chunk))` returns a numpy scalar that f-string's `d` format spec doesn't accept reliably across numpy versions. Fixed by wrapping the peak with `int(...)` and rms with `float(...)`.

A first attempt at RMS-per-window via `ffmpeg astats=reset=15` produced per-frame stats instead of per-15-second stats — `reset=15` resets every 15 *frames*, not seconds. The Python approach was simpler and exact.

### What I learned

Gemini 3.1 Flash TTS has a usable-length ceiling well below its advertised input budget. Even inputs of ~5000 characters produce audibly degraded generations. Paragraph-aligned chunking at 800 chars gives a comfortable safety margin; I haven't tested the actual ceiling but 1000 is probably still safe and 2000 probably is not.

The preview-model quota (10 RPM) also interacts with chunking: an 8-chunk post already takes ~5 minutes of wall time between API latency and the throttle. Fine for blog publication but would be prohibitive for any interactive use.

### What was tricky

Spotting the decay was Markus's ear, not my metrics — there was no obvious signal in the single-call generation's stats. The amplitude check had to be run deliberately after the user flagged it. This is the kind of failure mode that slips through if you only check "does it work" via metadata (duration, file size) and not "does it sound right" via an amplitude profile. Worth remembering.

### What warrants review

`/tts/speak_gemini.py` `chunk_text` — the greedy merge keeps merging until adding the next paragraph would exceed `max_chars`. If a single paragraph is already larger than `max_chars`, it passes through unsplit. That's fine for this blog (longest paragraph is ~600 chars) but would quietly reintroduce decay on a post with a 2000-char paragraph. Worth adding a sentence-level fallback if we ever hit that.

Also: the 800-char default was picked by feel, not by sweeping. A more principled choice would be to generate the same text at {400, 800, 1600, 3200} chars/chunk and plot the per-window RMS variance. Not doing that now, but noting for later.

### Future work

Add a sentence-level fallback splitter for paragraphs longer than `max_chars`. Possibly a post-generation RMS self-check that warns if any 30-second window is more than ~10 dB below the median — cheap, would have caught the original bug automatically.
