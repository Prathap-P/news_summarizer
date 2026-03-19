# Website Summarizer — Copilot Instructions

## Project Overview
A Flask web app that ingests news articles (URL) or YouTube videos (URL), condenses content via LLM map-reduce, generates a TTS-safe spoken-word audio file using Kokoro, and delivers it via email or Telegram.

## Architecture

```
app.py                         # Flask API — routes, global conversation state
main.py                        # read_website_content() via NewsURLLoader
youtube_transcript_fetcher.py  # get_youtube_transcript() — direct fetch(), no Whisper
whisper_transcriber.py         # Whisper pipeline: duration check, yt-dlp download, mlx-whisper
condenser_service.py           # Map-reduce LLM condensation pipeline with checkpoint resume
condensation_cache.py          # Checkpoint manager — atomic JSON, 24h TTL, resume support
llm_models.py                  # All LLM instances; get_model() factory
system_prompts.py              # All prompt strings (news, YouTube, map/reduce)
utils.py                       # remove_thinking_tokens(), backup file helpers
kokoro_tts.py                  # generate_audio(), create_audio_file()
email_sender.py                # send_email_with_audio/attachments
telegram_sender.py             # send_telegram_with_audio/attachments
templates/index.html           # Single-page frontend with Transcript + Audio queues
backup_content/                # Files saved when delivery fails
condensation_cache/            # Pipeline checkpoint JSON files (gitignored)
kokoro_outputs/                # Generated .wav files
yt_audio/                      # Audio downloaded by yt-dlp for Whisper transcription (gitignored)
```

## Key Conventions

### LLM Usage
- All LLM instances live in `llm_models.py`. Use `get_model("model_key")` everywhere else — never instantiate `ChatOpenAI`/`ChatGroq` inline.
- Local models connect to **LM Studio** at `http://localhost:1234/v1`; check with `check_llm_server()` before requests.
- Local model API key is always the dummy string `"test"`.
- Default model: `mlx_community_qwen_stream_local_llm`.

### TTS-Safe Output
- System prompts enforce **no markdown, no code blocks, no URLs, no bullet symbols**.
- Acronyms must be expanded on first use; numbers written in natural-reading form.
- All LLM responses are piped through `remove_thinking_tokens()` in `utils.py` before TTS. This function expects `<final_script>...</final_script>` tags around the model's final output.
- `remove_thinking_tokens()` returns `(text, False)` if tags are missing — always check the boolean and log a `[WARNING]` before continuing.

### YouTube Transcript Strategy — Two Hard-Separated Paths

The frontend has two queue types per category. Each maps to a distinct backend path with **no crossover**:

```
📋 Transcript Queue  →  fetch_mode = "transcript"
   get_youtube_transcript(url)
     └─ YouTubeTranscriptApi().fetch(video_id) — direct call, no list_transcripts(),
        no Whisper, no audio download.
        Returns "Error: ..." string if no transcript exists → user must use Audio Queue.

🎵 Audio Queue  →  fetch_mode = "audio"
   get_transcript_via_whisper(url, video_id)  [from whisper_transcriber.py]
     1. get_video_duration() — metadata only (no extractor_args override)
     2. warn if > 90 min
     3. download_audio() → yt_audio/<video_id>.mp3
        └─ REUSES cached file if it already exists and has non-zero size
     4. transcribe_audio() → mlx-whisper large-v3
        └─ GPU memory released in finally block (Apple Silicon only)
```

**Important rules:**
- `youtube_transcript_fetcher.py` never imports or calls `get_transcript_via_whisper`. Whisper belongs exclusively to the audio path.
- `app.py` normalises every YouTube URL to bare `https://www.youtube.com/watch?v=ID` form **before** any processing — playlist params (`&list=`, `&index=`, `&pp=`), tracking params, and all other query params are stripped.
- `fetch()` returns `FetchedTranscriptSnippet` objects (attribute access: `entry.text`) since `youtube-transcript-api >= 0.7`. Access `.text` attribute, not `["text"]` dict key.

### yt-dlp Client Strategy (2026-03+)
**Do NOT set `extractor_args` or `player_client` overrides.** As of yt-dlp 2026.03, the previously used clients are broken:
- `ios`, `mweb`, `android` — require a GVS PO Token; without it all media formats are skipped (only thumbnail storyboards returned → "Requested format is not available").
- `tv_embedded` — marked unsupported in yt-dlp ≥ 2026.03.

yt-dlp's automatic client selection (`android_vr` fallback as of 2026.03.13) resolves DASH audio formats (139/140/249/251) without any token. Let it select automatically.

Format string used: `"140/251/139/bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"`

### Condensation Pipeline (`condenser_service.py` + `condensation_cache.py`)
- Map phase: splits content with `RecursiveCharacterTextSplitter`, summarises each chunk individually.
- Reduce phase: batches `REDUCE_BATCH_SIZE = 3` chunks; consolidates if total > `FINAL_CONSOLIDATION_THRESHOLD = 15000` chars.
- Always stream: `for chunk in model.stream(input)` → accumulate → `remove_thinking_tokens()`.
- **Full checkpoint resume**: every MAP chunk and REDUCE batch is saved atomically after success. A crash loses at most one step. Chunks are stored before any LLM calls so resume uses identical splits.
- Cache key: `SHA-256(canonical_url | model_key | fetch_mode)[:16]`. YouTube variants all collapse to `yt:<video_id>`. News URLs strip tracking params.
- `fetch_mode` is part of the cache key — a Whisper-forced audio run never reuses a cached transcript-API run for the same video.
- TTL: 24 hours. Expired checkpoints purged at startup via `purge_expired_checkpoints()`.
- `condensation_cache/` and `yt_audio/` are gitignored.
- **Crash-safe streaming**: all 4 `current_model.stream()` loops (MAP, single-batch REDUCE, multi-batch REDUCE, final consolidation) are wrapped in `try/except Exception`. On crash: logs the error, increments the correct checkpoint retry counter (`map_retry_counts[str_idx]`, `reduce_retry_counts[key]`, or `consolidation_retries`), calls `_save()`, then raises `ValueError`. This converts silent model crashes (e.g. LM Studio `Exit code: null`) into recoverable checkpointed errors that `app.py`'s `except ValueError` block returns as 422 with `resume_progress`.

### URL Normalisation (YouTube)
In `app.py`'s `load_content` route, before any checkpoint or I/O work:
```python
if mode == 'youtube':
    _vid = extract_video_id(url)
    url = f"https://www.youtube.com/watch?v={_vid}"
```
This single normalisation point ensures the cache key, stored checkpoint `url` field, yt-dlp call, and transcript API call all use a stable identical form. Playlist URLs like `watch?v=ID&list=...&index=...` are reduced to `watch?v=ID`.

### Frontend Queue Design (`templates/index.html`)
Each of the three category panels (tech / social / science) has two queues:
- **📋 Transcript Queue** (`${cat}QueueList`) — sends `fetch_mode: "transcript"`
- **🎵 Audio Queue** (`${cat}AudioQueueList`) — sends `fetch_mode: "audio"`

Queue state is persisted to `localStorage`. Key behaviours:
- A URL is removed from the queue and `saveQueuesToStorage()` is called **immediately at dequeue time** (not after processing) so a page reload cannot replay already-handed-off URLs.
- Finished list entries show `[T]` or `[A]` prefix, plus `[CACHED:VIDEO_ID]` when a full cache hit was served.
- `/load_content` response includes `from_cache` (bool) and `video_id` (string | null).
- Failed list entries are written as `[T] <url> - Error: <message>` or `[A] <url> - Error: <message>`.

### Retry All Failed Button (`retryAllFailed()` in `templates/index.html`)
A single fixed button (bottom-right) handles both retry paths:
1. **Re-queue failed URLs**: scans all three `${cat}FailedList` textareas, parses `[T]`/`[A]` prefixes via regex `/^\[(A|T)\]\s+(https?:\/\/\S+)\s*-\s*Error:/`, pushes each URL back into the correct queue textarea (`QueueList` for `[T]`, `AudioQueueList` for `[A]`). Deduplicates within a single retry action using a `Set` keyed by `url|category|fetch_mode`. Calls `saveQueuesToStorage()` **before** the network call.
2. **Retry Telegram backups**: calls `POST /retry_failed_telegrams` (unchanged backend) to re-send `backup_content/` files.
3. Toast shows unified result: `"X URL(s) re-queued | Telegrams — ✅ Sent: N, ❌ Failed: N"`. Shows `"Nothing to retry"` when both paths find nothing.
- Failed list entries are **not** cleared by the retry — normal success/fail flow adds new FinishedList/FailedList entries on re-processing.

### Whisper Memory Management
`transcribe_audio()` always releases GPU memory in a `finally` block:
```python
finally:
    gc.collect()
    if _IS_APPLE_SILICON:          # guarded — no-op on Linux/CUDA
        mx.metal.clear_cache()
```
This frees ~3 GB of MLX buffers after each transcription instead of holding them for the Flask process lifetime.

### Logging Style
```python
print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] <message>")
print(f"[DEBUG]   <message>")
print(f"[WARNING] <message>")
print(f"[ERROR]   <message>")
```
Use structured `[LEVEL]` prefixes consistently. Do not use the `logging` module.

### Environment Variables
| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq cloud LLM |
| `GMAIL_ADDRESS` | Sender Gmail |
| `GMAIL_APP_PASSWORD` | Gmail app-specific password |
| `RECIPIENT_GMAIL_ADDRESS` | Recipient address |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Recipient chat ID (legacy fallback) |
| `TELEGRAM_CHANNEL_TECH` | Tech channel ID |
| `TELEGRAM_CHANNEL_SOCIAL` | Social channel ID |
| `TELEGRAM_CHANNEL_SCIENCE` | Science channel ID |
| `TELEGRAM_CHAT_ID_TECH` | Tech discussion group ID |
| `TELEGRAM_CHAT_ID_SOCIAL` | Social discussion group ID |
| `TELEGRAM_CHAT_ID_SCIENCE` | Science discussion group ID |

Load with `load_dotenv()` at module top, then `os.getenv("KEY")`. Never hardcode credentials.

### Global State in `app.py`
- `conversation_chain`, `session_history`, `current_model`, `current_model_key`, and `window_memory_100` are module-level globals (single-user dev tool — acceptable here).
- `current_model_key` is tracked as a string so the cache key can include the model name.
- Reset `window_memory_100` explicitly between sessions to avoid context bleed.
- Two chain styles coexist during refactor: legacy `ConversationChain` and new `RunnableWithMessageHistory`. Prefer the latter for new routes.

### Backup System
When delivery fails, `create_backup_file()` saves a `.txt` under `backup_content/`. Parse with `parse_backup_file()`; enumerate with `list_backup_files()`.

## Build & Run

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate
pip install -e .        # reads pyproject.toml (single source of truth for deps)

# Start Flask dev server (LM Studio must be running on port 1234 first)
python app.py

# CLI conversational interface
python news_reader.py
```

Add new packages to `pyproject.toml` only — do not create a separate `requirements.txt`.

## Pitfalls
- **LM Studio must be running on port 1234** before `app.py` starts if using local models.
- **Kokoro TTS** needs CUDA or Apple Silicon MPS; CPU fallback is very slow.
- YouTube transcript failures return error strings, not exceptions — always check output before passing to LLM.
- `ConversationBufferWindowMemory` is global; reset between sessions or context bleeds across users.
- **Never call `list_transcripts()`** in the transcript path — it fails frequently (bot detection, region locks, private videos). Use `fetch()` directly.
- **`FetchedTranscriptSnippet` objects** (youtube-transcript-api ≥ 0.7): use `entry.text`, not `entry["text"]`.
- **yt-dlp `extractor_args`**: do not set `player_client` overrides — ios/mweb/android all require GVS PO Tokens as of 2026-03 and will return zero media formats.
- Whisper (`mlx-whisper`) downloads `mlx-community/whisper-large-v3-mlx` (~3 GB) from HuggingFace on first use; subsequent runs use the cached model at `~/.cache/huggingface/hub/`.
- `yt-dlp` requires **ffmpeg** to be installed system-wide for the audio post-processor (`FFmpegExtractAudio`). Install with `brew install ffmpeg`.
- For videos >90 min, `get_transcript_via_whisper()` logs a warning but proceeds — transcription can take 5–15 min on Apple Silicon with `large-v3`.
- Downloaded audio is saved permanently to `yt_audio/<video_id>.mp3` and **reused on retry** — no re-download if the file exists with non-zero size.
- Checkpoint files expire after 24 hours and are purged at startup. Delete a checkpoint manually to force a full reprocess.
- YouTube playlist URLs must be normalised to `watch?v=ID` at the `load_content` entry point — never pass raw playlist URLs to yt-dlp or the transcript API.

---

## General Coding Standards

### Code Quality
- Clean, readable, production-ready Python
- Meaningful names; small, single-responsibility functions
- Type hints on all function signatures
- Error handling at system boundaries (user input, external APIs)

### Security
- Secrets via environment variables only — never hardcoded
- Validate and sanitise all external inputs (URLs, user text)
- No credentials or sensitive data in logs

### Python Best Practices
- Context managers for file/resource handling
- Prefer list comprehensions when readable
- `pathlib.Path` over raw string paths
- Docstrings on public functions; inline comments only for non-obvious logic

### Response Format
- Provide complete, runnable code
- Include install commands when introducing new dependencies
- Comments only where logic is non-obvious