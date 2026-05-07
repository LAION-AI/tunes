"""Evaluate multimodal (audio+text) models on the v2 annotation song set.

For each song in ``input/v2/annotations_export_2026-04-20.json`` (deduplicated by
``song_id``) the script:

  1. Resolves the original audio URL via the ``input/annotation-subset/*.jsonl``
     files using the ``<source>_<id>`` prefix convention.
  2. Sends an OpenAI v1 compatible ``chat/completions`` request with the audio
     URL attached and asks the model to produce a JSON judgement in the same
     shape as our human annotation records.
  3. Writes the judgement to ``input/models/<sanitized-model>.json`` after every
     song (atomic rename) so interrupted runs can be resumed: already-annotated
     ``song_id`` values are skipped on the next invocation.

Usage::

    uv run python analysis/model_eval.py                 # every entry in model_config.py
    uv run python analysis/model_eval.py --model <id>    # filter by model_id
    uv run python analysis/model_eval.py --limit 10      # quick smoke test

Environment (``analysis/.env``)::

    TRANSFORMERS_API_KEY=...
    TRANSFORMERS_API_ENDPOINT=https://.../v1/chat/completions
    HYPRLAB_API_KEY=...
    HYPRLAB_API_ENDPOINT=https://.../v1/chat/completions
    # Per-model provider/hyperparameters/thinking mode are read from model_config.py
"""

from __future__ import annotations

import argparse
import ast
import base64
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import urllib3
from dotenv import load_dotenv

# Suppress InsecureRequestWarning for local SGLang servers with self-signed certs.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE = Path(__file__).resolve().parent
SUBSET_DIR = HERE / "input" / "annotation-subset"
ANN_FILE = HERE / "input" / "v2" / "annotations_export_2026-04-20.json"
V2_MINUS_V1_DIR = HERE / "input" / "v2-v1"
OUT_DIR = HERE / "input" / "models"
MODEL_CONFIG_FILE = HERE / "model_config.py"
CACHE_DIR = HERE / ".cache"
CACHE_CROPPED_DIR = HERE / ".cache_cropped"

# Snippet policy mirrors the annotation web app: AI (non-human) songs are
# cropped to a deterministic 29s window; human songs are played in full.
SNIPPET_DURATION_SEC = 29
SNIPPET_UNKNOWN_DURATION_MAX_START_SEC = 31
_V2_MINUS_V1_SONG_IDS: Optional[set[str]] = None

# song_source -> (jsonl filename, id field, ordered candidate audio url fields)
SOURCE_CONFIG: dict[str, tuple[str, str, list[str]]] = {
    "suno":    ("suno_subset.jsonl",    "id",   ["audio_url"]),
    "udio":    ("udio_subset.jsonl",    "id",   ["song_path"]),
    "mureka":  ("mureka_subset.jsonl",  "uuid", ["audio_url"]),
    "sonauto": ("sonauto_subset.jsonl", "id",   ["song_path"]),
    "human":   ("human_songs.jsonl",    "uuid", ["audio_url"]),
}

ALLOWED_AI_ASPECTS = [
    "Singing voice",
    "Instrument sounds",
    "Lyrics",
    "Rhythm",
    "Harmony",
    "Overall composition",
    "Everything",
]

CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5
POSTHOC_UNCERTAIN_MAX_CONFIDENCE = 2

SYSTEM_PROMPT = (
    "You are a careful music perception judge. You will listen to a single "
    "music track and evaluate it as if you were a human annotator in a "
    "blinded perceptual study. Base every judgement on what you actually "
    "hear in the audio, not on external knowledge about the track, the "
    "artist, or the generation platform."
)

USER_INSTRUCTIONS = f"""Listen to the attached audio and evaluate it.

Return ONLY a single JSON object (no markdown fences, no commentary) that
exactly matches this schema:

{{
  "authenticity_assessment": "real" | "ai-generated",
  "authenticity_confidence": integer 1-5,
  "familiarity_level": string chosen from ["Yes, I know this song", "Sounds familiar", "Never heard it", "Uncertain"],
  "aesthetic_quality": integer 1-10,
  "playlist_likelihood": integer 1-10,
  "musical_creativity": integer 1-10,
  "production_quality": integer 1-10,
  "emotional_engagement": integer 1-10,
  "ai_aspects": array of strings (suggested: {ALLOWED_AI_ASPECTS}, or add custom free text) OR null,
  "mood_tags": array of short mood strings (suggested: "Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness", "Power", "Joyful activation", "Tension", "Sadness", or free text),
  "aesthetic_comment": string or null,
  "song_description": string or null
}}

Rules:
- "authenticity_assessment" is a forced binary choice: always choose either
  "real" or "ai-generated". Never output "uncertain".
- "authenticity_confidence" is your confidence in that forced choice:
  1 = very unsure, 2 = somewhat unsure, 3 = mixed/neutral,
  4 = fairly confident, 5 = very confident.
- Rate each of the five dimensions on an integer 1 (worst) to 10 (best) scale.
- Provide "ai_aspects" (non-empty list) ONLY when authenticity_assessment is
  "ai-generated"; otherwise set it to null.
- "mood_tags" may be an empty list.
- Keep "aesthetic_comment" and "song_description" under 30 words or null.
- Output must be valid JSON parseable by ``json.loads``.
"""


def build_user_instructions(model_id: str, thinking_mode: str) -> str:
    """Build user prompt with optional explicit reasoning-format request."""
    mode = (thinking_mode or "").strip().lower()
    if is_gemini_model(model_id):
        if mode in {"low", "medium", "high"}:
            return (
                "Before the final JSON answer, include a concise reasoning "
                "summary prefixed exactly with `thought\\n`.\n\n"
                + USER_INSTRUCTIONS
            )
        return USER_INSTRUCTIONS
    # MOSS/SGLang reasoning is returned via reasoning_content in the response;
    # no special prompt prefix needed.
    if is_moss_model(model_id):
        return USER_INSTRUCTIONS
    if mode in {"on", "true", "enabled", "think"}:
        return (
            "Thinking mode is enabled. First output your reasoning in the exact "
            "format `<|channel>thought\\n...<channel|>`, then output the final "
            "JSON answer.\n\n"
            + USER_INSTRUCTIONS
        )
    return USER_INSTRUCTIONS


def load_model_configs(path: Path) -> list[dict[str, Any]]:
    """Load model run configurations from ``model_config.py``."""
    raw = path.read_text(encoding="utf-8")
    data = ast.literal_eval(raw)
    if not isinstance(data, list):
        raise ValueError("model_config.py must contain a list of dicts")

    configs: list[dict[str, Any]] = []
    for i, cfg in enumerate(data):
        if not isinstance(cfg, dict):
            raise ValueError(f"Config index {i} is not a dict")
        model_id = str(cfg.get("model_id", "")).strip()
        if not model_id:
            raise ValueError(f"Config index {i} missing model_id")
        raw_thinking_mode = cfg.get("thinking_mode", "off")
        if isinstance(raw_thinking_mode, bool):
            # Backward compatibility with older configs.
            thinking_mode = "on" if raw_thinking_mode else "off"
        else:
            thinking_mode = str(raw_thinking_mode).strip().lower() or "off"

        configs.append({
            "model_id": model_id,
            "temperature": float(cfg.get("temperature", 1.0)),
            "top_p": float(cfg.get("top_p", 0.95)),
            "top_k": int(cfg.get("top_k", 64)),
            "max_tokens": int(cfg.get("max_tokens", 4096)),
            "thinking_mode": thinking_mode,
            "provider": str(cfg.get("provider", "transformers")).strip().lower(),
        })
    return configs


def model_run_name(cfg: dict[str, Any]) -> str:
    think = f"think-{cfg.get('thinking_mode', 'off')}"
    return (
        f"{cfg['model_id']}__temp-{cfg['temperature']}"
        f"__top-p-{cfg['top_p']}__top-k-{cfg['top_k']}"
        f"__max-tok-{cfg['max_tokens']}__{think}"
    )


def is_gemini_model(model_id: str) -> bool:
    return "gemini" in model_id.lower()


def is_moss_model(model_id: str) -> bool:
    return "moss" in model_id.lower()


def resolve_gemini_thinking_level(thinking_mode: str) -> Optional[str]:
    mode = (thinking_mode or "").strip().lower()
    if mode in {"minimal", "low", "medium", "high"}:
        return mode
    if mode in {"off", "false", "none"}:
        return "minimal"
    if mode in {"on", "true", "default"}:
        return None  # keep provider default dynamic behavior
    return None


def resolve_reasoning_effort(thinking_mode: str) -> str:
    """Map thinking_mode to chat-completions reasoning_effort."""
    mode = (thinking_mode or "").strip().lower()
    if mode in {"high", "on", "true", "enabled", "think"}:
        return "high"
    if mode in {"medium"}:
        return "medium"
    # Treat minimal/off/unknown as low-latency effort.
    return "low"


def build_system_prompt(model_id: str, thinking_mode: str) -> str:
    base = SYSTEM_PROMPT
    # Gemini uses thinkingLevel in generationConfig, not chat control tokens.
    if is_gemini_model(model_id):
        return base
    # MOSS/SGLang thinking is controlled via separate_reasoning + chat_template_kwargs,
    # not prompt tokens.
    if is_moss_model(model_id):
        return base
    mode = (thinking_mode or "").strip().lower()
    # Gemma thinking is controlled only by presence of <|think|>.
    # Be strict: enable only for explicit on-like values.
    if mode in {"on", "true", "enabled", "think"}:
        # Gemma 3n/4 thinking trigger token.
        return "<|think|>\n" + base
    return base


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def load_subset_index() -> dict[str, dict[str, Any]]:
    """Return ``song_id -> {audio_url, title, ...}`` across all subset files."""
    index: dict[str, dict[str, Any]] = {}
    for source, (fname, id_field, url_fields) in SOURCE_CONFIG.items():
        path = SUBSET_DIR / fname
        if not path.exists():
            print(f"[warn] missing subset file: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get(id_field)
                if not sid:
                    continue
                url: Optional[str] = None
                for fld in url_fields:
                    if obj.get(fld):
                        url = obj[fld]
                        break
                index[f"{source}_{sid}"] = {
                    "audio_url": url,
                    "title": obj.get("title"),
                    "source": source,
                }
    return index


def load_unique_songs() -> list[dict[str, Any]]:
    """Return unique songs from the v2 annotations in deterministic order."""
    with ANN_FILE.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("song_id")
        if not sid or sid in seen:
            continue
        seen[sid] = {
            "song_id": sid,
            "song_source": rec.get("song_source"),
            "song_title": rec.get("song_title"),
            "song_duration": rec.get("song_duration"),
            "song_genres": rec.get("song_genres"),
            "song_moods": rec.get("song_moods"),
            "song_tags": rec.get("song_tags"),
        }
    # Sort for stable resume ordering.
    return [seen[k] for k in sorted(seen.keys())]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_existing_output(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] could not parse existing output {path}: {exc}",
              file=sys.stderr)
    return []


def parse_model_json(text: str) -> dict[str, Any]:
    """Best effort extraction of a JSON object from the model response."""
    stripped = text.strip()
    # Remove Gemma-style thought channel wrappers when present.
    stripped = re.sub(
        r"<\|channel\|>\s*thought.*?<\|channel\|>",
        "",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    stripped = re.sub(
        r"<\|channel>\s*thought.*?<channel\|>",
        "",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    # Strip common markdown code fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped,
                     flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    # Try raw parse first; raw_decode stops after the first valid JSON value,
    # tolerating any trailing text (e.g. two objects or a stray comment).
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    if start != -1:
        try:
            obj, _ = decoder.raw_decode(stripped, start)
            return obj
        except json.JSONDecodeError:
            pass
        # Some models emit double commas (e.g. `"key": 5,,`). Strip them and retry.
        cleaned = re.sub(r",(\s*,)+", ",", stripped)
        # Also remove trailing commas before closing braces/brackets.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        start2 = cleaned.find("{")
        if start2 != -1:
            try:
                obj, _ = decoder.raw_decode(cleaned, start2)
                return obj
            except json.JSONDecodeError:
                pass
    debug_path = HERE / "DEBUG.txt"
    with debug_path.open("a", encoding="utf-8") as _dbg:
        _dbg.write("--- unparseable response ---\n")
        _dbg.write(text)
        _dbg.write("\n--- end ---\n\n")
    raise ValueError("Model response did not contain parseable JSON")


def resolve_provider_credentials(provider: str) -> tuple[str, str]:
    """Return (api_key, endpoint) using <PROVIDER>_API_KEY/_API_ENDPOINT env vars."""
    prefix = provider.strip().upper()
    key = os.getenv(f"{prefix}_API_KEY", "").strip()
    endpoint = os.getenv(f"{prefix}_API_ENDPOINT", "").strip()
    if not key or not endpoint:
        raise ValueError(
            f"Missing credentials for provider '{provider}'. "
            f"Expected {prefix}_API_KEY and {prefix}_API_ENDPOINT in .env"
        )
    return key, endpoint


def derive_google_api_urls(base_endpoint: str, model_id: str) -> tuple[str, str]:
    """Build (upload_url, generate_url) from provider endpoint."""
    parsed = urlparse(base_endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    upload_url = f"{origin}/upload/v1beta/files"
    generate_url = f"{origin}/v1beta/models/{model_id}:generateContent"
    return upload_url, generate_url


def derive_gemini_native_url(base_endpoint: str, model_id: str) -> str:
    """Derive ``https://host/v1beta/models/<model>:generateContent`` from a
    provider endpoint such as Hyprlab's ``/v1/chat/completions``. This is
    needed because the OpenAI-compat chat endpoint silently drops inline
    audio for Gemini models, while the native generateContent endpoint
    correctly processes ``inline_data`` audio parts."""
    parsed = urlparse(base_endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    override = os.getenv("HYPRLAB_GEMINI_NATIVE_BASE", "").strip()
    if override:
        origin = override.rstrip("/")
    return f"{origin}/v1beta/models/{model_id}:generateContent"


def detect_mime_type(audio_url: str, content_type: Optional[str]) -> str:
    if content_type:
        c = content_type.split(";")[0].strip().lower()
        # Treat generic binary types as unknown and fall through to URL detection.
        if c and c not in {"application/octet-stream", "binary/octet-stream", "application/binary"}:
            return c
    lower = audio_url.lower()
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".ogg"):
        return "audio/ogg"
    if lower.endswith(".m4a"):
        return "audio/mp4"
    return "audio/mpeg"


def normalize_audio_inline_format(mime_type: str, audio_url: str) -> str:
    """Map mime/url hints to provider-accepted input_audio format tokens."""
    m = (mime_type or "").strip().lower()
    if m in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    if m in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return "wav"
    if m in {"audio/ogg", "audio/vorbis"}:
        return "ogg"
    if m in {"audio/mp4", "audio/x-m4a", "audio/m4a"}:
        return "m4a"
    # Fallback from URL extension if mime is noisy/unknown.
    lower = audio_url.lower()
    if lower.endswith(".wav"):
        return "wav"
    if lower.endswith(".ogg"):
        return "ogg"
    if lower.endswith(".m4a") or lower.endswith(".mp4"):
        return "m4a"
    return "mp3"


def is_non_human_song(song: dict[str, Any]) -> bool:
    """Return True for AI-generated songs (matching the annotation web app)."""
    source = str(song.get("song_source") or "").strip().lower()
    if source:
        return source != "human"
    song_id = str(song.get("song_id") or "").strip().lower()
    return not song_id.startswith("human_")


def load_v2_minus_v1_song_ids() -> set[str]:
    """Load song IDs contained in input/v2-v1/annotations_export_*.json."""
    global _V2_MINUS_V1_SONG_IDS
    if _V2_MINUS_V1_SONG_IDS is not None:
        return _V2_MINUS_V1_SONG_IDS

    files = sorted(V2_MINUS_V1_DIR.glob("annotations_export_*.json"))
    if not files:
        print(
            f"[warn] no v2-v1 annotation file found in {V2_MINUS_V1_DIR}; "
            "cropping disabled",
            file=sys.stderr,
        )
        _V2_MINUS_V1_SONG_IDS = set()
        return _V2_MINUS_V1_SONG_IDS

    path = files[-1]
    try:
        with path.open("r", encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[warn] failed to load {path}: {exc}; cropping disabled",
            file=sys.stderr,
        )
        _V2_MINUS_V1_SONG_IDS = set()
        return _V2_MINUS_V1_SONG_IDS

    ids: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                sid = str(row.get("song_id") or "")
                if sid:
                    ids.add(sid)
    _V2_MINUS_V1_SONG_IDS = ids
    return _V2_MINUS_V1_SONG_IDS


def should_crop_song(song: dict[str, Any]) -> bool:
    """Crop only non-human songs that belong to the v2-v1 subset."""
    if not is_non_human_song(song):
        return False
    sid = str(song.get("song_id") or "")
    return sid in load_v2_minus_v1_song_ids()


def print_song_stats(songs: list[dict[str, Any]]) -> None:
    """Print a short overview of song split and crop policy."""
    total_songs = len(songs)
    ai_songs = [song for song in songs if is_non_human_song(song)]
    cropped_ai_songs = [song for song in ai_songs if should_crop_song(song)]
    uncropped_ai_songs = len(ai_songs) - len(cropped_ai_songs)
    human_songs = total_songs - len(ai_songs)

    print("Total songs:", total_songs)
    print("AI songs:", len(ai_songs))
    print("Uncropped AI songs:", uncropped_ai_songs)
    print("Cropped AI songs:", len(cropped_ai_songs))
    print("Real (human) songs:", human_songs)
    print("# Comment: Real (human) songs are only 30s each anyway")


def deterministic_random(song_id: str) -> float:
    """djb2 hash -> float in [0, 1), byte-identical to the annotation platform.

    Mirrors the JS snippet ``hash = ((hash << 5) + hash + c) | 0`` by using
    ``ctypes.c_int32`` so large strings produce the same 32-bit signed
    truncation as JavaScript.
    """
    h = 5381
    for ch in song_id:
        h = ctypes.c_int32((h << 5) + h + ord(ch)).value
    return abs(h) % 10000 / 10000.0


def get_snippet_start_sec(song_id: str, duration_ms: int) -> int:
    """Return the snippet start offset in seconds, matching the annotation app.

    If the song duration is known (>0 ms), the playable window is the first
    two thirds of the song minus the 29s snippet length; otherwise a fixed
    31s window is used so the snippet finishes by ~60s.
    """
    rand = deterministic_random(song_id)
    if duration_ms and duration_ms > 0:
        song_duration_sec = duration_ms / 1000.0
        max_start = max(0.0, (song_duration_sec * 2.0 / 3.0) - SNIPPET_DURATION_SEC)
        return math.floor(rand * max_start) if max_start > 0 else 0
    return math.floor(rand * SNIPPET_UNKNOWN_DURATION_MAX_START_SEC)


def _coerce_duration_ms(value: Any) -> int:
    """Best-effort parse of song_duration into integer milliseconds."""
    try:
        if value is None:
            return 0
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def _extension_for_mime(mime_type: str, audio_url: str) -> str:
    m = (mime_type or "").strip().lower()
    if "wav" in m:
        return ".wav"
    if "ogg" in m:
        return ".ogg"
    if "mp4" in m or "m4a" in m:
        return ".m4a"
    if "mpeg" in m or "mp3" in m:
        return ".mp3"
    lower = audio_url.lower()
    for ext in (".mp3", ".wav", ".ogg", ".m4a", ".mp4"):
        if lower.endswith(ext):
            return ".m4a" if ext == ".mp4" else ext
    return ".mp3"


def _run_ffmpeg_crop(src: Path, dst: Path, start_sec: int, duration_sec: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg binary not found on PATH; install ffmpeg to enable "
            "non-human song cropping"
        )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(src),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg crop failed ({proc.returncode}): {proc.stderr[:500]}"
        )


def get_inference_audio(
    song: dict[str, Any],
    audio_url: str,
    timeout: int,
) -> tuple[bytes, str]:
    """Return (bytes, mime) ready to send to a model.

    Songs are returned unchanged unless they are non-human AND in v2-v1.
    Matching songs are cropped with ffmpeg to a deterministic 29s snippet
    matching the annotation web app and cached under
    ``analysis/.cache_cropped/`` keyed by song metadata.
    """
    CACHE_CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    song_id = str(song.get("song_id") or "")
    source = str(song.get("song_source") or "").strip().lower() or "unknown"
    duration_ms = _coerce_duration_ms(song.get("song_duration"))

    key_src = (
        f"{source}|{song_id}|{duration_ms}|{SNIPPET_DURATION_SEC}|{audio_url}"
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    bin_path = CACHE_CROPPED_DIR / f"{key}.bin"
    meta_path = CACHE_CROPPED_DIR / f"{key}.json"

    if bin_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mime = detect_mime_type(audio_url, str(meta.get("mime_type", "")).strip() or None)
            return bin_path.read_bytes(), mime
        except (json.JSONDecodeError, OSError):
            pass

    full_bytes, full_mime = get_cached_audio(audio_url, timeout)

    if not should_crop_song(song):
        bin_path.write_bytes(full_bytes)
        meta_path.write_text(
            json.dumps(
                {
                    "audio_url": audio_url,
                    "song_id": song_id,
                    "song_source": source,
                    "mime_type": full_mime,
                    "size_bytes": len(full_bytes),
                    "cropped": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return full_bytes, full_mime

    start_sec = get_snippet_start_sec(song_id, duration_ms)
    src_path = CACHE_CROPPED_DIR / f"{key}.src{_extension_for_mime(full_mime, audio_url)}"
    tmp_out = CACHE_CROPPED_DIR / f"{key}.out.mp3"
    try:
        src_path.write_bytes(full_bytes)
        _run_ffmpeg_crop(src_path, tmp_out, start_sec, SNIPPET_DURATION_SEC)
        cropped_bytes = tmp_out.read_bytes()
    finally:
        for p in (src_path, tmp_out):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    mime_out = "audio/mpeg"
    bin_path.write_bytes(cropped_bytes)
    meta_path.write_text(
        json.dumps(
            {
                "audio_url": audio_url,
                "song_id": song_id,
                "song_source": source,
                "mime_type": mime_out,
                "size_bytes": len(cropped_bytes),
                "cropped": True,
                "start_sec": start_sec,
                "duration_sec": SNIPPET_DURATION_SEC,
                "orig_duration_ms": duration_ms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return cropped_bytes, mime_out


def get_cached_audio(audio_url: str, timeout: int) -> tuple[bytes, str]:
    """Return audio bytes + mime type, caching by URL hash in analysis/.cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(audio_url.encode("utf-8")).hexdigest()
    bin_path = CACHE_DIR / f"{key}.bin"
    meta_path = CACHE_DIR / f"{key}.json"

    if bin_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mime = detect_mime_type(audio_url, str(meta.get("mime_type", "")).strip() or None)
            return bin_path.read_bytes(), mime
        except (json.JSONDecodeError, OSError):
            # Corrupt cache entry: fall through to refetch.
            pass

    resp = requests.get(audio_url, timeout=timeout)
    resp.raise_for_status()
    audio_bytes = resp.content
    mime_type = detect_mime_type(audio_url, resp.headers.get("Content-Type"))

    bin_path.write_bytes(audio_bytes)
    meta_path.write_text(
        json.dumps(
            {
                "audio_url": audio_url,
                "mime_type": mime_type,
                "size_bytes": len(audio_bytes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return audio_bytes, mime_type


def call_model_hyprlab_gemini(
    *,
    endpoint: str,
    api_key: str,
    model_cfg: dict[str, Any],
    song: dict[str, Any],
    audio_url: str,
    timeout: int,
) -> tuple[str, int]:
    """Gemini-native ``generateContent`` flow via Hyprlab with inline audio.

    Hyprlab's OpenAI-compat ``v1/chat/completions`` endpoint does not
    reliably forward the audio payload to Gemini models (the server replies
    as if no audio was attached). The native ``v1beta/models/<id>:generateContent``
    endpoint accepts ``contents[].parts[].inline_data`` with the raw bytes
    and works across all supported audio MIME types (mp3, m4a, ogg, wav).
    """
    model = model_cfg["model_id"]
    thinking_mode = model_cfg.get("thinking_mode", "off")
    system_prompt = build_system_prompt(model, thinking_mode)
    user_instructions = build_user_instructions(model, thinking_mode)
    started = time.monotonic()

    audio_bytes, mime_type = get_inference_audio(song, audio_url, timeout)
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise RuntimeError(
            f"Audio too large for inline mode ({len(audio_bytes)} bytes > 20MB)"
        )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    inline_mime = mime_type if "/" in (mime_type or "") else "audio/mpeg"

    thinking_level = resolve_gemini_thinking_level(thinking_mode)
    thinking_cfg: dict[str, Any] = {"includeThoughts": True}
    if thinking_level is not None:
        thinking_cfg["thinkingLevel"] = thinking_level

    generate_url = derive_gemini_native_url(endpoint, model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_instructions},
                    {"inline_data": {"mime_type": inline_mime, "data": audio_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": model_cfg["temperature"],
            "topP": model_cfg["top_p"],
            "topK": model_cfg["top_k"],
            "maxOutputTokens": model_cfg["max_tokens"],
            "thinkingConfig": thinking_cfg,
        },
    }
    resp = requests.post(generate_url, headers=headers, json=payload,
                         timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        debug_path = HERE / "DEBUG.txt"
        with debug_path.open("a", encoding="utf-8") as _dbg:
            _dbg.write("--- unparseable API response (gemini-native) ---\n")
            _dbg.write(resp.text)
            _dbg.write("\n--- end ---\n\n")
        raise

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(
            f"No candidates in Gemini response: {str(data)[:300]}"
        )
    parts = (candidates[0].get("content") or {}).get("parts") or []
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        txt = p.get("text")
        if not txt:
            continue
        if p.get("thought") is True:
            thought_chunks.append(txt)
        else:
            answer_chunks.append(txt)
    answer = "".join(answer_chunks).strip()
    if not answer:
        raise RuntimeError(
            f"No text parts in Gemini response: {str(data)[:300]}"
        )
    if thought_chunks:
        thought_joined = "".join(thought_chunks).strip()
        content = f"thought\n{thought_joined}\n\n{answer}".strip()
    else:
        content = answer

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return content, elapsed_ms


def call_model_moss(
    *,
    endpoint: str,
    api_key: str,
    model_cfg: dict[str, Any],
    song: dict[str, Any],
    audio_url: str,
    timeout: int,
) -> tuple[str, int]:
    """SGLang/MOSS /v1/chat/completions flow with audio_url and separate_reasoning.

    Audio is always downloaded to the local cache first and sent as a base64
    data-URL so the SGLang server never needs to reach remote audio CDNs.
    SSL certificate verification is disabled because the local SGLang server
    typically runs with a self-signed cert.
    """
    model = model_cfg["model_id"]
    thinking_mode = model_cfg.get("thinking_mode", "off")
    user_instructions = build_user_instructions(model, thinking_mode)
    started = time.monotonic()

    # Always fetch audio locally (uses .cache / .cache_cropped) and send
    # inline so the SGLang server never needs to reach remote audio CDNs.
    audio_bytes, mime_type = get_inference_audio(song, audio_url, timeout)
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise RuntimeError(
            f"Audio too large for inline mode ({len(audio_bytes)} bytes > 20MB)"
        )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_ref = f"data:{mime_type or 'audio/mpeg'};base64,{audio_b64}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_ref},
                    },
                    {"type": "text", "text": user_instructions},
                ],
            }
        ],
        "max_tokens": model_cfg["max_tokens"],
        "temperature": model_cfg["temperature"],
        "top_p": model_cfg["top_p"],
        "separate_reasoning": True,
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        debug_path = HERE / "DEBUG.txt"
        with debug_path.open("a", encoding="utf-8") as _dbg:
            _dbg.write("--- unparseable API response (moss) ---\n")
            _dbg.write(resp.text)
            _dbg.write("\n--- end ---\n\n")
        raise

    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = str(content or "").strip()
    reasoning_content = str(msg.get("reasoning_content", "") or "").strip()
    if not content:
        raise RuntimeError("No text found in MOSS chat/completions response")
    if reasoning_content:
        content = f"thought\n{reasoning_content}\n\n{content}".strip()

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return content, elapsed_ms


def call_model_hyprlab(
    *,
    endpoint: str,
    api_key: str,
    model_cfg: dict[str, Any],
    song: dict[str, Any],
    audio_url: str,
    timeout: int,
) -> tuple[str, int]:
    """Hyprlab chat/completions flow with inline audio + reasoning_effort."""
    model = model_cfg["model_id"]
    reasoning_effort = resolve_reasoning_effort(model_cfg.get("thinking_mode", "off"))
    system_prompt = build_system_prompt(model, model_cfg["thinking_mode"])
    user_instructions = build_user_instructions(model, model_cfg["thinking_mode"])
    started = time.monotonic()

    audio_bytes, mime_type = get_inference_audio(song, audio_url, timeout)
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise RuntimeError(
            f"Audio too large for inline mode ({len(audio_bytes)} bytes > 20MB)"
        )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_format = normalize_audio_inline_format(mime_type, audio_url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": model_cfg["temperature"],
        "max_tokens": model_cfg["max_tokens"],
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "text"},
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instructions},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": audio_format,
                        },
                    },
                ],
            },
        ],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        data = resp.json()
    except json.JSONDecodeError:
        debug_path = HERE / "DEBUG.txt"
        with debug_path.open("a", encoding="utf-8") as _dbg:
            _dbg.write("--- unparseable API response (hyprlab-chat) ---\n")
            _dbg.write(resp.text)
            _dbg.write("\n--- end ---\n\n")
        raise
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    content = str(content or "").strip()
    reasoning_content = str(msg.get("reasoning_content", "") or "").strip()
    if not content:
        raise RuntimeError("No text found in chat/completions response")
    if reasoning_content:
        content = f"thought\n{reasoning_content}\n\n{content}".strip()

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return content, elapsed_ms


def coerce_int(value: Any, lo: int = 1, hi: int = 10) -> Optional[int]:
    try:
        if value is None:
            return None
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def build_record(
    *,
    song: dict[str, Any],
    parsed: dict[str, Any],
    model_id: str,
    session_id: str,
    duration_ms: int,
    raw_response: str,
) -> dict[str, Any]:
    auth = parsed.get("authenticity_assessment")
    if auth not in {"real", "ai-generated"}:
        # Backward compatibility for older prompt variants that could return
        # "uncertain": map to a low-confidence forced binary guess.
        if auth == "uncertain":
            auth = "ai-generated" if parsed.get("ai_aspects") else "real"
        else:
            auth = "real"

    confidence = coerce_int(
        parsed.get("authenticity_confidence"),
        lo=CONFIDENCE_MIN,
        hi=CONFIDENCE_MAX,
    )
    if confidence is None:
        # If missing, treat as maximally uncertain to preserve comparability.
        confidence = CONFIDENCE_MIN

    posthoc_auth = (
        "uncertain"
        if confidence <= POSTHOC_UNCERTAIN_MAX_CONFIDENCE
        else auth
    )

    ai_aspects = parsed.get("ai_aspects")
    if auth != "ai-generated":
        ai_aspects = None
    elif isinstance(ai_aspects, list):
        ai_aspects = json.dumps(ai_aspects, ensure_ascii=False)
    elif ai_aspects in (None, ""):
        ai_aspects = None
    else:
        ai_aspects = json.dumps([str(ai_aspects)], ensure_ascii=False)

    mood = parsed.get("mood_tags")
    if isinstance(mood, list):
        mood_str = json.dumps(mood, ensure_ascii=False)
    elif mood is None:
        mood_str = "[]"
    else:
        mood_str = json.dumps([str(mood)], ensure_ascii=False)

    return {
        "annotation_id": str(uuid.uuid4()),
        "session_id": session_id,
        "participant_id": f"model:{model_id}",
        "song_id": song["song_id"],
        "song_source": song["song_source"],
        "authenticity_assessment": auth,
        "authenticity_confidence": confidence,
        "authenticity_assessment_posthoc_uncertain": posthoc_auth,
        "authenticity_posthoc_uncertain_threshold": (
            POSTHOC_UNCERTAIN_MAX_CONFIDENCE
        ),
        "familiarity_level": parsed.get("familiarity_level") or "Uncertain",
        "aesthetic_quality": coerce_int(parsed.get("aesthetic_quality")),
        "playlist_likelihood": coerce_int(parsed.get("playlist_likelihood")),
        "musical_creativity": coerce_int(parsed.get("musical_creativity")),
        "production_quality": coerce_int(parsed.get("production_quality")),
        "emotional_engagement": coerce_int(parsed.get("emotional_engagement")),
        "ai_aspects": ai_aspects,
        "mood_tags": mood_str,
        "aesthetic_comment": parsed.get("aesthetic_comment") or None,
        "song_description": parsed.get("song_description") or None,
        "annotation_duration_ms": int(duration_ms),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat()
                        .replace("+00:00", "Z"),
        "participant_age": None,
        "participant_age_range": None,
        "participant_musical_genres": None,
        "participant_musical_engagement": "model",
        "participant_formal_training_years": None,
        "participant_listening_device": "api",
        "participant_listening_context": "model",
        "participant_environment": "model",
        "participant_ai_music_experience": "model",
        "song_title": song.get("song_title"),
        "song_duration": song.get("song_duration"),
        "song_genres": song.get("song_genres"),
        "song_moods": song.get("song_moods"),
        "song_tags": song.get("song_tags"),
        "model_id": model_id,
        "model_raw_response": raw_response,
    }


def call_model(
    *,
    endpoint: str,
    api_key: str,
    model_cfg: dict[str, Any],
    song: dict[str, Any],
    audio_url: str,
    timeout: int,
    max_retries: int,
) -> tuple[str, int]:
    if model_cfg.get("provider") == "hyprlab":
        if is_gemini_model(model_cfg["model_id"]):
            return call_model_hyprlab_gemini(
                endpoint=endpoint,
                api_key=api_key,
                model_cfg=model_cfg,
                song=song,
                audio_url=audio_url,
                timeout=timeout,
            )
        return call_model_hyprlab(
            endpoint=endpoint,
            api_key=api_key,
            model_cfg=model_cfg,
            song=song,
            audio_url=audio_url,
            timeout=timeout,
        )

    if model_cfg.get("provider") == "moss":
        return call_model_moss(
            endpoint=endpoint,
            api_key=api_key,
            model_cfg=model_cfg,
            song=song,
            audio_url=audio_url,
            timeout=timeout,
        )

    # For songs that must be cropped, we ship the cropped bytes in-line
    # (URL-based variants would let the remote server fetch the full song and
    # defeat the snippet policy). All other songs keep the prior URL behaviour.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    model = model_cfg["model_id"]
    user_instructions = build_user_instructions(model, model_cfg["thinking_mode"])

    audio_variants: list[tuple[str, dict[str, Any], bool]]
    if should_crop_song(song):
        audio_bytes, mime_type = get_inference_audio(song, audio_url, timeout)
        if len(audio_bytes) > 20 * 1024 * 1024:
            raise RuntimeError(
                f"Cropped audio too large for inline mode ({len(audio_bytes)} bytes > 20MB)"
            )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_format = normalize_audio_inline_format(mime_type, audio_url)
        data_url = f"data:{mime_type or 'audio/mpeg'};base64,{audio_b64}"
        audio_variants = [
            (
                "input-audio-inline",
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": audio_format},
                },
                False,
            ),
            (
                "audio-url-data",
                {"type": "audio_url", "audio_url": {"url": data_url}},
                False,
            ),
            ("native-audio-inline", {"type": "audio", "audio": data_url}, False),
        ]
    else:
        # Warm the full-song cache for observability and parity with before.
        get_cached_audio(audio_url, timeout)
        audio_variants = [
            ("native-audio", {"type": "audio", "audio": audio_url}, False),
            ("audio-url-object", {"type": "audio_url",
                                  "audio_url": {"url": audio_url}}, False),
            ("audio-url-string", {"type": "audio_url", "audio_url": audio_url}, False),
            # Some servers require content as plain string and accept an extra
            # audio field directly on the user message.
            ("message-audio-url", {"audio_url": audio_url}, True),
            ("message-audio-url-object", {"audio_url": {"url": audio_url}}, True),
        ]

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        if _STOP:
            raise KeyboardInterrupt("Stop requested")
        for variant_name, audio_part, content_as_string in audio_variants:
            started = time.monotonic()
            user_msg: dict[str, Any]
            if content_as_string:
                user_msg = {
                    "role": "user",
                    "content": user_instructions,
                    **audio_part,
                }
            else:
                user_msg = {
                    "role": "user",
                    "content": [
                        audio_part,
                        {"type": "text", "text": user_instructions},
                    ],
                }
            payload = {
                "model": model,
                "temperature": model_cfg["temperature"],
                "top_p": model_cfg["top_p"],
                "top_k": model_cfg["top_k"],
                "max_tokens": model_cfg["max_tokens"],
                "messages": [
                    {
                        "role": "system",
                        "content": build_system_prompt(
                            model,
                            model_cfg["thinking_mode"]
                        ),
                    },
                    user_msg,
                ],
            }
            try:
                r = requests.post(endpoint, headers=headers, json=payload,
                                  timeout=timeout)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                if r.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    # Some servers return content as list of parts.
                    content = "".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict)
                    )
                return content, elapsed_ms
            except Exception as exc:  # noqa: BLE001 -- bubble after retries
                last_exc = exc
                # For schema errors, immediately try next audio variant.
                if "HTTP 422" in str(exc):
                    print(f"[warn] variant {variant_name} rejected (422), "
                          "trying next format...", file=sys.stderr)
                    continue
                if _STOP:
                    raise KeyboardInterrupt("Stop requested") from exc
                # Non-schema failure: stop variant cycling this attempt.
                break

        if last_exc is not None and _STOP:
            raise KeyboardInterrupt("Stop requested") from last_exc
        backoff = min(30, 2 ** attempt)
        print(f"[warn] attempt {attempt}/{max_retries} failed: {last_exc}; "
              f"retrying in {backoff}s", file=sys.stderr)
        for _ in range(backoff * 10):
            if _STOP:
                raise KeyboardInterrupt("Stop requested") from last_exc
            time.sleep(0.1)
    assert last_exc is not None
    raise last_exc


_STOP = False
_SIGINT_COUNT = 0


def _install_sigint() -> None:
    def handler(signum, frame):  # noqa: ARG001
        global _STOP, _SIGINT_COUNT
        _SIGINT_COUNT += 1
        _STOP = True
        if _SIGINT_COUNT == 1:
            print("\n[info] interrupt received, stopping after current in-flight "
                  "request (press Ctrl+C again to force quit now)...",
                  file=sys.stderr)
            return
        raise KeyboardInterrupt("Forced stop requested")
    signal.signal(signal.SIGINT, handler)


def run_model(
    *,
    model_cfg: dict[str, Any],
    songs: list[dict[str, Any]],
    url_index: dict[str, dict[str, Any]],
    limit: Optional[int],
    timeout: int,
    max_retries: int,
    overwrite: bool,
    workers: int = 1,
) -> None:
    provider = model_cfg.get("provider", "transformers")
    api_key, endpoint = resolve_provider_credentials(provider)
    run_name = model_run_name(model_cfg)
    model = model_cfg["model_id"]
    out_path = OUT_DIR / f"{sanitize_model_name(run_name)}.json"
    existing = [] if overwrite else load_existing_output(out_path)
    done_ids = {r.get("song_id") for r in existing if r.get("song_id")}

    pending = [s for s in songs if s["song_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    print(f"[{model}] provider: {provider}")
    print(f"[{model}] output: {out_path}")
    print(f"[{model}] already done: {len(done_ids)} / {len(songs)}; "
          f"pending this run: {len(pending)}")
    if workers > 1:
        print(f"[{model}] concurrent workers: {workers}")

    session_id = str(uuid.uuid4())
    results = list(existing)

    if workers > 1:
        # --- Concurrent path (for SGLang/MOSS and similar servers) ---
        #
        # Pre-fetch all audio sequentially first so that worker threads only
        # read from the on-disk cache and never race on ffmpeg cropping.
        print(f"[{model}] pre-fetching audio for {len(pending)} songs...")
        fetchable = []
        for song in pending:
            if _STOP:
                break
            sid = song["song_id"]
            meta = url_index.get(sid)
            if not meta or not meta.get("audio_url"):
                print(f"[{model}] {sid}: no audio URL, skipping",
                      file=sys.stderr)
                continue
            try:
                get_inference_audio(song, meta["audio_url"], timeout)
                fetchable.append(song)
            except Exception as exc:  # noqa: BLE001
                print(f"[{model}] prefetch {sid}: {exc}", file=sys.stderr)

        if _STOP:
            print(f"[{model}] done. total records: {len(results)}")
            return

        print(f"[{model}] audio ready; submitting {len(fetchable)} "
              f"inference requests ({workers} workers)...")

        lock = threading.Lock()

        def _process_song(song: dict[str, Any]) -> Optional[dict[str, Any]]:
            if _STOP:
                return None
            sid = song["song_id"]
            meta = url_index.get(sid)
            audio_url = meta["audio_url"]  # guaranteed present after prefetch
            print(f"[{model}] -> {sid} ({audio_url[:80]})")
            raw, elapsed_ms = call_model(
                endpoint=endpoint,
                api_key=api_key,
                model_cfg=model_cfg,
                song=song,
                audio_url=audio_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            parsed = parse_model_json(raw)
            rec = build_record(
                song=song,
                parsed=parsed,
                model_id=model,
                session_id=session_id,
                duration_ms=elapsed_ms,
                raw_response=raw,
            )
            rec["model_run_name"] = run_name
            rec["model_temperature"] = model_cfg["temperature"]
            rec["model_top_p"] = model_cfg["top_p"]
            rec["model_top_k"] = model_cfg["top_k"]
            rec["model_max_tokens"] = model_cfg["max_tokens"]
            rec["model_thinking_mode"] = model_cfg["thinking_mode"]
            return rec

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_song = {
                pool.submit(_process_song, song): song
                for song in fetchable
                if not _STOP
            }
            total = len(future_to_song)
            done_count = 0
            for future in as_completed(future_to_song):
                song = future_to_song[future]
                sid = song["song_id"]
                try:
                    rec = future.result()
                except KeyboardInterrupt:
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[{model}] {sid}: failed: {exc}", file=sys.stderr)
                    continue
                finally:
                    done_count += 1
                if rec is not None:
                    with lock:
                        results.append(rec)
                        atomic_write_json(out_path, results)
                    print(f"[{model}] ({done_count}/{total}) {sid} done")
                if _STOP:
                    for f in future_to_song:
                        f.cancel()
                    break

    else:
        # --- Sequential path (all other providers) ---
        for i, song in enumerate(pending, start=1):
            if _STOP:
                break
            sid = song["song_id"]
            meta = url_index.get(sid)
            if not meta or not meta.get("audio_url"):
                print(f"[{model}] ({i}/{len(pending)}) {sid}: no audio URL, "
                      "skipping", file=sys.stderr)
                continue
            audio_url = meta["audio_url"]
            print(f"[{model}] ({i}/{len(pending)}) {sid} -> {audio_url[:80]}")
            try:
                raw, elapsed_ms = call_model(
                    endpoint=endpoint,
                    api_key=api_key,
                    model_cfg=model_cfg,
                    song=song,
                    audio_url=audio_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                parsed = parse_model_json(raw)
                record = build_record(
                    song=song,
                    parsed=parsed,
                    model_id=model,
                    session_id=session_id,
                    duration_ms=elapsed_ms,
                    raw_response=raw,
                )
            except KeyboardInterrupt:
                print(f"[{model}] interrupt acknowledged; stopping now.",
                      file=sys.stderr)
                break
            except Exception as exc:  # noqa: BLE001 -- log & continue
                print(f"[{model}] {sid}: failed: {exc}", file=sys.stderr)
                continue

            results.append(record)
            results[-1]["model_run_name"] = run_name
            results[-1]["model_temperature"] = model_cfg["temperature"]
            results[-1]["model_top_p"] = model_cfg["top_p"]
            results[-1]["model_top_k"] = model_cfg["top_k"]
            results[-1]["model_max_tokens"] = model_cfg["max_tokens"]
            results[-1]["model_thinking_mode"] = model_cfg["thinking_mode"]
            atomic_write_json(out_path, results)

    print(f"[{model}] done. total records: {len(results)}")


def main() -> int:
    global ANN_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None,
                        help="filter configured model_id values (may repeat)")
    parser.add_argument("--config", default=str(MODEL_CONFIG_FILE),
                        help="path to model config file (python list literal)")
    parser.add_argument("--annotations", default=None,
                        help="path to the v2 annotations JSON "
                             f"(default: {ANN_FILE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of songs per model (debug)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="per-request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="per-song retry count on network/API errors")
    parser.add_argument("--overwrite", action="store_true",
                        help="ignore existing output file and restart")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent inference workers (default: 8 for moss, "
                             "1 for all other providers)")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")

    all_model_cfgs = load_model_configs(Path(args.config))
    if args.model:
        allow = {m.strip() for m in args.model if m.strip()}
        model_cfgs = [c for c in all_model_cfgs if c["model_id"] in allow]
    else:
        model_cfgs = all_model_cfgs
    if not model_cfgs:
        print("No model configs selected. Check --model/--config.",
              file=sys.stderr)
        return 2

    if args.annotations:
        ANN_FILE = Path(args.annotations)

    url_index = load_subset_index()
    songs = load_unique_songs()
    print(f"loaded {len(songs)} unique songs from {ANN_FILE.name}")
    print_song_stats(songs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _install_sigint()

    try:
        for model_cfg in model_cfgs:
            if _STOP:
                break
            provider = model_cfg.get("provider", "transformers")
            if args.workers is not None:
                workers = args.workers
            else:
                workers = 16 if provider == "moss" else 1
            run_model(
                model_cfg=model_cfg,
                songs=songs,
                url_index=url_index,
                limit=args.limit,
                timeout=args.timeout,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
                workers=workers,
            )
    except KeyboardInterrupt:
        print("\n[info] interrupted by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
