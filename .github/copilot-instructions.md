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
audio_config.py                # ASR/TTS backend selection via env vars
kokoro_tts.py                  # generate_audio(), create_audio_file() — Kokoro backend
qwen_omni_backend.py           # generate_audio_qwen(), get_transcript_via_qwen() — Qwen2.5-Omni backend
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

### ASR / TTS Backend Selection (`audio_config.py`)

The ASR and TTS backends are configurable via environment variables — no code change needed to switch:

| Variable | Default | Options |
|---|---|---|
| `ASR_BACKEND` | `qwen_omni` | `qwen_omni`, `whisper` |
| `TTS_BACKEND` | `qwen_omni` | `qwen_omni`, `kokoro` |
| `QWEN_OMNI_MODEL_ID` | `Qwen/Qwen2.5-Omni-3B` | Any HF model ID |
| `QWEN_OMNI_SPEAKER` | `Chelsie` | `Chelsie` (female), `Ethan` (male) |

`app.py` imports the correct backend at startup based on these values:
```python
from audio_config import ASR_BACKEND, TTS_BACKEND
if ASR_BACKEND == "qwen_omni":
    from qwen_omni_backend import get_transcript_via_qwen as get_transcript_via_whisper
else:
    from whisper_transcriber import get_transcript_via_whisper

if TTS_BACKEND == "qwen_omni":
    from qwen_omni_backend import generate_audio_qwen as generate_audio
    from kokoro_tts import create_audio_file
else:
    from kokoro_tts import generate_audio, create_audio_file
```

All call sites (`generate_audio()`, `create_audio_file()`, `get_transcript_via_whisper()`) remain unchanged — the aliasing at import time is the entire switching mechanism.

**`create_audio_file()` is always imported from `kokoro_tts`** regardless of `TTS_BACKEND` — it is backend-agnostic (accepts any 24 kHz float32 numpy array and writes a `.wav` file).

### Qwen2.5-Omni Backend (`qwen_omni_backend.py`)

- **Model**: `Qwen/Qwen2.5-Omni-3B` (default). Any `Qwen2.5-Omni-*` variant is supported via `QWEN_OMNI_MODEL_ID`.
- **Single model for both ASR and TTS** — loaded once, reused for both paths via module-level lazy singleton (`_model`, `_processor`).
- **Lazy loading**: model is NOT loaded at import time. First call to `generate_audio_qwen()` or `get_transcript_via_qwen()` triggers `_get_model()`.
- **Apple Silicon**: loads with `device_map={"":"mps"}` and `torch_dtype=torch.float16`. On other platforms: `device_map="auto"`, `torch_dtype="auto"`.
- **Processor**: `Qwen2_5OmniProcessor.from_pretrained(model_id, use_fast=True)`.

**TTS path** (`generate_audio_qwen`):
- Required system prompt: `"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."`
- Every message's `content` must be a **list of dicts** (`[{"type": "text", "text": "..."}]`), including the system message — never a bare string.
- Call: `model.generate(**inputs, use_audio_in_video=False, speaker=QWEN_OMNI_SPEAKER)`
- Audio extracted via: `out.waveform if hasattr(out, "waveform") else out[1]`
- Returns 24 kHz float32 numpy array — same contract as Kokoro's `generate_audio()`.

**ASR path** (`get_transcript_via_qwen`):
- Reuses `download_audio()` and `get_video_duration()` from `whisper_transcriber.py` (lazy import inside function).
- Call: `model.generate(**inputs, use_audio_in_video=False, return_audio=False)`
- Decode: `processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)` where `ids = output.sequences if hasattr(output, "sequences") else output`
- Returns plain text string or `"Error: ..."` on failure — same contract as `get_transcript_via_whisper()`.

**Memory management**: both functions release MPS memory via `torch.mps.empty_cache()` in a `finally` block. GPU tensors are explicitly `del`-ed before the `finally` runs so the cache clear is effective.

**`process_mm_info`** from `qwen_omni_utils` is imported at module level (not per-call). If the package is missing, an `ImportError` with a clear message is raised at import time.

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
- Uses `model.invoke(input)` (not streaming) → `response.content` → `remove_thinking_tokens()`.
- **Full checkpoint resume**: every MAP chunk and REDUCE batch is saved atomically after success. A crash loses at most one step. Chunks are stored before any LLM calls so resume uses identical splits.
- Cache key: `SHA-256(canonical_url | model_key | fetch_mode)[:16]`. YouTube variants all collapse to `yt:<video_id>`. News URLs strip tracking params.
- `fetch_mode` is part of the cache key — a Whisper-forced audio run never reuses a cached transcript-API run for the same video.
- TTL: 24 hours. Expired checkpoints purged at startup via `purge_expired_checkpoints()`.
- `condensation_cache/` and `yt_audio/` are gitignored.
- **Crash-safe invoke**: all 4 `current_model.invoke()` calls (MAP, single-batch REDUCE, multi-batch REDUCE, final consolidation) are wrapped in `try/except Exception`. On crash: logs the error, increments the correct checkpoint retry counter (`map_retry_counts[str_idx]`, `reduce_retry_counts[key]`, or `consolidation_retries`), calls `_save()`, then raises `ValueError`. This converts silent model crashes (e.g. LM Studio `Exit code: null`) into recoverable checkpointed errors that `app.py`'s `except ValueError` block returns as 422 with `resume_progress`.
- `streaming=True` / `stream_usage=True` flags are commented out on all local LLM model definitions in `llm_models.py` — do not re-enable them for the condenser models.

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
- **Failed list is a card UI** — not a textarea. Each failed item is stored as `{ uid, url, error, fetch_mode, title }` in `failedData[category]` (a module-level JS object), persisted to `localStorage` as a JSON array under `${cat}Failed`. Cards show a YouTube thumbnail (`img.youtube.com/vi/{ID}/hqdefault.jpg`), video title (fetched once via oEmbed, cached in the item), error text, and per-card **↺ Retry** / **✕ Dismiss** buttons. News URLs show a 📰 placeholder instead. Do not read `.value` from a `${cat}FailedList` element — it is a `<div>`, not a textarea.

### Retry All Failed Button (`retryAllFailed()` in `templates/index.html`)
A single fixed button (bottom-right) handles both retry paths:
1. **Re-queue failed URLs**: iterates `failedData[category]` directly (no textarea parsing), pushes each URL back into the correct queue textarea (`QueueList` for transcript, `AudioQueueList` for audio). Deduplicates within a single retry action using a `Set` keyed by `url|category|fetch_mode`. Calls `saveQueuesToStorage()` **before** the network call.
2. **Retry Telegram backups**: calls `POST /retry_failed_telegrams` (unchanged backend) to re-send `backup_content/` files.
3. Toast shows unified result: `"X URL(s) re-queued | Telegrams — ✅ Sent: N, ❌ Failed: N"`. Shows `"Nothing to retry"` when both paths find nothing.
- Failed cards are **not** cleared by the retry — use the per-card ↺ Retry or ✕ Dismiss buttons to remove individual items.

### Whisper Memory Management
`transcribe_audio()` in `whisper_transcriber.py` always releases GPU memory in a `finally` block:
```python
finally:
    gc.collect()
    if _IS_APPLE_SILICON:          # guarded — no-op on Linux/CUDA
        mx.metal.clear_cache()     # MLX Metal pool (mlx-whisper)
```
This frees ~3 GB of MLX buffers after each transcription.

The Qwen Omni backend uses an equivalent pattern with `torch.mps.empty_cache()` (PyTorch MPS pool) instead of `mx.metal.clear_cache()` — the two models use separate Metal memory pools.

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
| `ASR_BACKEND` | `qwen_omni` (default) or `whisper` |
| `TTS_BACKEND` | `qwen_omni` (default) or `kokoro` |
| `QWEN_OMNI_MODEL_ID` | HuggingFace model ID (default: `Qwen/Qwen2.5-Omni-3B`) |
| `QWEN_OMNI_SPEAKER` | TTS voice: `Chelsie` (default, female) or `Ethan` (male) |
| `DEFAULT_MODEL_KEY` | Startup LLM key from `models_collection` (default: `mlx_community_qwen_stream_local_llm`) |
| `LM_STUDIO_BASE_URL` | LM Studio OpenAI-compatible endpoint (default: `http://localhost:1234/v1`) |
| `GROQ_MODEL_ID` | Groq model ID (default: `openai/gpt-oss-20b`) |
| `GEMMA_MODEL_ID` | Gemma local model ID (default: `google/gemma-3-27b`) |
| `NEMOTRON_MODEL_ID` | Nemotron local model ID (default: `nvidia/nemotron-3-nano`) |
| `NEMOTRON_STREAM_MODEL_ID` | Nemotron streaming model ID (default: `nvidia/nemotron-3-nano`) |
| `NEXVERIDIAN_QWEN_MODEL_ID` | Nexveridian Qwen model ID (default: `nexveridian/qwen3.5-35b-a3b`) |
| `MLX_QWEN_MODEL_ID` | MLX Qwen model ID (default: `mlx-community/qwen3.5-35b-a3b`) |
| `DEEPSEEK_MODEL_ID` | DeepSeek model ID (default: `deepseek/deepseek-r1-0528-qwen3-8b`) |
| `GPT_OSS_MODEL_ID` | GPT OSS model ID (default: `openai/gpt-oss-20b`) |
| `MISTRAL_MODEL_ID` | Mistral model ID (default: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`) |
| `KOKORO_LANG_CODE` | Kokoro language code (default: `a` = American English) |
| `KOKORO_VOICE` | Kokoro TTS voice name (default: `af_sarah`) |
| `WHISPER_MODEL_ID` | Whisper model ID (default: `mlx-community/whisper-large-v3-mlx`) |

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

**New dependencies added for Qwen Omni backend** (already in `pyproject.toml`):
- `accelerate==1.13.0` — required by `transformers` for `device_map`
- `qwen-omni-utils==0.0.9` — `process_mm_info()` helper
- `torchvision==0.26.0` — required by `Qwen2_5OmniProcessor` even for audio-only use
- `soxr==1.0.0` — audio resampling dependency of `qwen-omni-utils`
- `torch==2.11.0`, `torchaudio==2.11.0` — upgraded from 2.9.1 to satisfy torchvision

## Pitfalls
- **LM Studio must be running on port 1234** before `app.py` starts if using local models.
- **Kokoro TTS** needs CUDA or Apple Silicon MPS; CPU fallback is very slow.
- **Kokoro + PyTorch 2.11+** emits a `UserWarning` about tensor resize (`resized since it had shape []`). This is a Kokoro `0.9.4` internal bug — audio is generated correctly. It is suppressed with `warnings.filterwarnings("ignore", ...)` scoped to the generator loop in `kokoro_tts.py`.
- **Qwen2.5-Omni** model (~7 GB) downloads from HuggingFace on first use to `~/.cache/huggingface/hub/`. LM Studio does NOT need to be running for the Qwen backend.
- **Qwen Omni system message `content` must be a list of dicts**, not a bare string. `process_mm_info` iterates content items by dict key — a plain string causes `TypeError: string indices must be integers`.
- **`process_mm_info` and `model.generate()` both need `use_audio_in_video=False`** for audio-only / text-only inputs. Missing it from either call causes the model's video decoder to activate and fail.
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