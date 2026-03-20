"""Audio download and Whisper transcription pipeline.

Flow:
    get_transcript_via_whisper(url, video_id)
        ├─ get_video_duration()   — metadata-only fetch, warn if > 90 min
        ├─ download_audio()       — yt-dlp → yt_audio/<id>.mp3
        │       └─ REUSES cached file if it already exists and is non-empty
        └─ transcribe_audio()     — mlx-whisper large-v3 in isolated subprocess
                └─ All MLX/Metal unified memory reclaimed when subprocess exits
"""

import gc
import multiprocessing as mp
import platform
from datetime import datetime
from pathlib import Path

import yt_dlp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YT_AUDIO_DIR = Path("yt_audio")
YT_AUDIO_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DURATION_WARNING_THRESHOLD_SECONDS = 90 * 60

# Platform guard: Metal memory cache only exists on Apple Silicon.
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# ---------------------------------------------------------------------------
# yt-dlp client strategy (2026-03+)
#
# Do NOT specify extractor_args / player_client overrides.
#
# History of why explicit client pinning was removed:
#   ios         — Now requires a GVS PO Token; without it every audio/video
#                 format is skipped, leaving only thumbnail storyboards.
#   mweb        — Same; also requires GVS PO Token since early 2026.
#   tv_embedded — Marked unsupported in yt-dlp ≥ 2026.03.
#   android     — Returns only one combined format (360p) without PO token.
#
# yt-dlp's built-in automatic client selection (android_vr fallback as of
# 2026.03.13) resolves all DASH audio formats (139/140/249/251) without any
# cookie or PO token.  Let it do its job.
# ---------------------------------------------------------------------------

# Stable YouTube DASH audio format IDs in preference order:
#   140 = AAC  128k m4a  — present on virtually every public video since 2013
#   251 = Opus 160k webm — present on most modern videos
#   139 = AAC   48k m4a  — fallback for age-restricted or older videos
# Generic quality-based selectors follow as last resort.
_AUDIO_FORMAT = "140/251/139/bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _release_gpu_memory() -> None:
    """Release MLX/Metal GPU memory in the current process.

    No-op on non-Apple-Silicon platforms so the code runs unmodified on Linux/CUDA.
    """
    gc.collect()
    if _IS_APPLE_SILICON:
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception as e:
            print(f"[WARNING] Could not clear MPS cache: {e}")
    print(f"[INFO]    [{_ts()}] Whisper GPU memory released")


def _transcribe_worker(
    audio_path_str: str,
    model: str,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker that runs Whisper inside an isolated subprocess.

    All MLX/Metal unified-memory allocations (~3–10 GB) are reclaimed
    unconditionally when this process exits — no GC or cache-clear needed.
    """
    try:
        import mlx_whisper  # imported inside subprocess so the parent never loads it

        result = mlx_whisper.transcribe(audio_path_str, path_or_hf_repo=model)
        text = result.get("text", "").strip()
        result_queue.put(("ok", text))
    except Exception as e:
        result_queue.put(("error", str(e)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_video_duration(url: str) -> int | None:
    """Return video duration in seconds without downloading any media.

    Uses yt-dlp's automatic client selection — no extractor_args override.
    Returns None on any error.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("duration")
    except Exception as e:
        print(f"[WARNING] Could not fetch video duration: {e}")
        return None


def download_audio(url: str, video_id: str) -> Path | None:
    """Download the audio track from a YouTube URL using yt-dlp.

    Saves to ``yt_audio/<video_id>.mp3``.

    Idempotent: if the file already exists and has a non-zero size it is reused
    without re-downloading. This means a crash mid-transcription does not force
    a full re-download on retry.

    Returns the resolved Path on success, None on failure.
    """
    output_path = YT_AUDIO_DIR / f"{video_id}.mp3"

    # Reuse cached file instead of re-downloading
    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(
            f"[INFO]    [{_ts()}] Reusing cached audio: "
            f"{output_path.name} ({size_mb:.1f} MB)"
        )
        return output_path

    output_template = str(YT_AUDIO_DIR / video_id)
    ydl_opts = {
        "format": _AUDIO_FORMAT,
        "outtmpl": output_template + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"[ERROR]   yt-dlp download failed: {e}")
        return None

    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"[ERROR]   Audio file missing or empty after download: {output_path}")
        return None

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(
        f"[INFO]    [{_ts()}] Audio downloaded: {output_path.name} ({size_mb:.1f} MB)"
    )
    return output_path


def transcribe_audio(audio_path: Path) -> str | None:
    """Transcribe an audio file with mlx-whisper (large-v3 by default).

    Runs Whisper inside a ``spawn``-ed subprocess so all MLX/Metal unified
    memory (~3–10 GB) is unconditionally reclaimed by the OS when the child
    process exits.  This is the only reliable way to free unified memory;
    ``gc.collect()`` + ``mx.metal.clear_cache()`` only clear the allocator's
    free-list and cannot release weights that are still referenced inside
    mlx_whisper's internal model cache.

    Returns plain transcript text, or None on failure or empty result.
    """
    print(
        f"[INFO]    [{_ts()}] Starting Whisper transcription: "
        f"{audio_path.name}  (model: {WHISPER_MODEL})"
    )

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    proc = ctx.Process(
        target=_transcribe_worker,
        args=(str(audio_path), WHISPER_MODEL, result_queue),
    )

    try:
        proc.start()
        proc.join()  # block until transcription finishes (no timeout — can be 5-15 min)

        if result_queue.empty():
            print(
                f"[ERROR]   Whisper subprocess exited without a result "
                f"(exit code: {proc.exitcode})"
            )
            return None

        status, value = result_queue.get_nowait()
        if status == "error":
            print(f"[ERROR]   Whisper transcription failed: {value}")
            return None

        text: str = value
        if not text:
            print("[WARNING] Whisper returned an empty transcript.")
            return None

        print(f"[INFO]    [{_ts()}] Whisper transcription complete: {len(text):,} chars")
        print(f"[INFO]    [{_ts()}] Whisper subprocess exited — unified memory reclaimed")
        return text

    except Exception as e:
        print(f"[ERROR]   Whisper transcription failed: {e}")
        return None
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        # Belt-and-suspenders: release anything the parent process may have touched
        _release_gpu_memory()


def get_transcript_via_whisper(url: str, video_id: str) -> str:
    """Full Whisper pipeline: duration check → audio download → transcription.

    Returns:
        Transcript text string on success.
        A string prefixed with ``"Error:"`` on any failure.
    """
    duration = get_video_duration(url)
    if duration is not None:
        minutes = duration // 60
        print(f"[INFO]    [{_ts()}] Video duration: {minutes} min ({duration}s)")
        if duration > DURATION_WARNING_THRESHOLD_SECONDS:
            print(
                f"[WARNING] Video is {minutes} min — "
                f"Whisper transcription may take 5–15 min on large-v3"
            )

    print(f"[INFO]    [{_ts()}] Downloading audio via yt-dlp...")
    audio_path = download_audio(url, video_id)
    if not audio_path:
        return "Error: Failed to download audio from YouTube."

    transcript = transcribe_audio(audio_path)
    if not transcript:
        return "Error: Whisper transcription failed or produced empty output."

    return transcript
