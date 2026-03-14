# Website Summarizer — Copilot Instructions

## Project Overview
A Flask web app that ingests news articles (URL) or YouTube videos (URL), condenses content via LLM map-reduce, generates a TTS-safe spoken-word audio file using Kokoro, and delivers it via email or Telegram.

## Architecture

```
app.py                         # Flask API — routes, global conversation state
main.py                        # read_website_content() via NewsURLLoader
youtube_transcript_fetcher.py  # get_youtube_transcript() via YouTubeTranscriptApi
condenser_service.py           # Map-reduce LLM condensation pipeline
llm_models.py                  # All LLM instances; get_model() factory
system_prompts.py              # All prompt strings (news, YouTube, map/reduce)
utils.py                       # remove_thinking_tokens(), backup file helpers
kokoro_tts.py                  # generate_audio(), create_audio_file()
email_sender.py                # send_email_with_audio/attachments
telegram_sender.py             # send_telegram_with_audio/attachments
templates/index.html           # Single-page frontend
backup_content/                # Files saved when delivery fails
kokoro_outputs/                # Generated .wav files
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

### Condensation Pipeline (`condenser_service.py`)
- Map phase: splits content with `RecursiveCharacterTextSplitter`, summarises each chunk individually.
- Reduce phase: batches `REDUCE_BATCH_SIZE = 3` chunks; consolidates if total > `FINAL_CONSOLIDATION_THRESHOLD = 15000` chars.
- Always stream: `for chunk in model.stream(input)` → accumulate → `remove_thinking_tokens()`.

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
| `TELEGRAM_CHAT_ID` | Recipient chat ID |

Load with `load_dotenv()` at module top, then `os.getenv("KEY")`. Never hardcode credentials.

### Global State in `app.py`
- `conversation_chain`, `session_history`, `current_model`, and `window_memory_100` are module-level globals (single-user dev tool — acceptable here).
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