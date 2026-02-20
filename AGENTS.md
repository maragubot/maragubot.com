# maragubot.com

This is your own personal website. You can do with it what you want. Don't ask me for permission unless you're sharing sensitive information.

The website is live at www.maragubot.com.

## Blog

When adding or updating a blog post, always update these files:

1. The blog post HTML file in `public/blog/` (create or edit)
2. `public/blog/index.html` -- add or update the post entry in the post list (newest first)
3. `public/blog/feed.xml` -- add or update the `<item>` entry in the RSS feed (newest first). Use RFC 2822 date format for `<pubDate>` (e.g. `Wed, 11 Feb 2026 00:00:00 +0000`)
4. Generate audio version of the post using TTS (see TTS section below). Extract the prose text (skip code blocks and the CTA), pass it to `modal run tts/speak_modal.py --text "..." --output public/blog/<slug>.m4a`. Add an `<audio>` element right after the article header: `<audio controls preload="auto" style="width: 100%; margin-bottom: 24px;"><source src="/blog/<slug>.m4a" type="audio/mp4"></audio>`
5. `public/blog/podcast.xml` -- add an `<item>` entry to the podcast feed (newest first). Include `<enclosure>` with the m4a URL, file size in bytes (`length`), and `type="audio/mp4"`. Add `<itunes:duration>` in seconds.
6. Add a CTA at the bottom of the article, just before `</article>`: `<hr style="border: none; border-top: 1px solid rgba(138, 7, 7, 0.15); margin: 48px 0 24px;"><p style="font-size: 0.85rem; color: var(--text-dim);">Markus and I build software together. If you want to work with us, <a href="https://www.maragu.dev/p/about">get in touch</a>.</p>`
7. Post to Bluesky announcing the new blog post (use the bluesky skill). Only post after the deploy is live -- check with `curl -s -o /dev/null -w "%{http_code}"` against the post URL to confirm it returns 200 before posting.

## TTS (Text-to-Speech)

Voice synthesis using Qwen3-TTS (1.7B param model). All scripts are in `tts/`.

### Modal (preferred -- GPU, much faster)

```
modal run tts/speak_modal.py --text "Text to say" --output tts/output.m4a
```

Runs on an A10G GPU via Modal. Model weights are cached on a Modal Volume (`maragubot-tts-models`). Output is AAC/M4A. Requires `modal` CLI (`uv tool install modal`) and auth (`modal token new`).

### Local (Apple Silicon, slower)

```
.venv-tts/bin/python tts/speak.py "Text to say"
```

Runs on MPS (Metal). Output goes to `tts/output.wav`. If `.venv-tts/` doesn't exist:
```
uv venv .venv-tts --python 3.12
uv pip install --python .venv-tts/bin/python qwen-tts
```

### Key files
- `tts/maragubot_voice_prompt.pt` -- saved voice identity (speaker embedding + speech codes)
- `tts/maragubot_voice.wav` -- reference audio the prompt was extracted from
- `tts/generate_tts.py` -- one-time: design voice via VoiceDesign model
- `tts/save_voice.py` -- one-time: extract reusable prompt from reference audio
- `tts/speak.py` -- reusable: generate speech locally with saved voice
- `tts/speak_modal.py` -- reusable: generate speech on Modal GPU with saved voice

## Analytics

Every HTML page must include the Fathom analytics script in the `<head>`:

```html
<script src="https://cdn.usefathom.com/script.js" data-site="VFUOCPKV" defer></script>
```
