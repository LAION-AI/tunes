#!/usr/bin/env python3
"""
LAION-Tunes Search Server
==========================
FastAPI server providing semantic search, BM25 text search, music similarity
search, and metadata filtering over the LAION-Tunes music dataset.

Features:
  - Vector similarity search via FAISS (tag/lyric/mood/caption/transcription embeddings)
  - Music audio similarity search via Whisper encoder mean-pooled embeddings
  - BM25 text search (tags, caption, transcription, privacy-hashed lyrics)
  - Combined search: vector search + BM25 + filter by aesthetics/subset + re-rank
  - Real-time query embedding via EmbeddingGemma 300M
  - Audio upload → Whisper encoder embedding → FAISS similarity search
  - Dark-mode HTML frontend with audio players

Usage:
    python server.py [--port 7860] [--gpu 0] [--host 0.0.0.0]

Requires: search_index/ directory built by build_search_index.py
"""

import os
import sys
import json
import time
import hmac
import hashlib
import pickle
import re
import sqlite3
import logging
import argparse
import tempfile
import io
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import faiss
from fastapi import FastAPI, Query, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import BM25Index from the build script (needed for pickle deserialization)
sys.path.insert(0, str(Path(__file__).parent))
from build_search_index import BM25Index

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "search_index"
INSTRUMENTAL_INDEX_DIR = BASE_DIR / "results" / "instrumental_index"
HTML_PATH = BASE_DIR / "index.html"

LYRICS_HMAC_KEY = b"laion-tunes-search-2026-secret-key"

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ── Global state (loaded at startup) ─────────────────────────────────────────
state = {
    "faiss": {},       # field -> faiss.Index
    "id_maps": {},     # field -> numpy array of row_ids
    "bm25": {},        # field -> BM25Index
    "db_path": None,   # SQLite path
    "embedder": None,  # SentenceTransformer model
    "total_tracks": 0,
    # Whisper encoder for music similarity
    "whisper_processor": None,
    "whisper_encoder": None,
    "whisper_device": None,
    "whisper_rid_to_idx": {},  # row_id -> faiss_idx for reverse lookup
    "audio_emb_cache": {},     # (ip, filename) -> {"embedding": np.array, "expires": timestamp}
    # Instrumental subset indices
    "instrumental": {
        "faiss": {},       # field -> faiss.Index (whisper, caption)
        "id_maps": {},     # field -> numpy array
        "bm25": {},        # field -> BM25Index (caption, tags)
        "db_path": None,   # SQLite path
        "total_tracks": 0,
        "whisper_rid_to_idx": {},
    },
}

# ── Audio Embedding Cache ────────────────────────────────────────────────────
CACHE_TTL = 3600  # 1 hour
_cache_lock = threading.Lock()

def cache_get(ip, filename):
    """Get cached audio embedding. Returns numpy array or None."""
    key = (ip, filename)
    with _cache_lock:
        entry = state["audio_emb_cache"].get(key)
        if entry and entry["expires"] > time.time():
            return entry["embedding"]
        # Expired or missing
        if entry:
            del state["audio_emb_cache"][key]
        return None

def cache_set(ip, filename, embedding):
    """Cache an audio embedding."""
    key = (ip, filename)
    with _cache_lock:
        state["audio_emb_cache"][key] = {
            "embedding": embedding,
            "expires": time.time() + CACHE_TTL,
        }
        # Evict expired entries (max 1000 entries)
        if len(state["audio_emb_cache"]) > 1000:
            now = time.time()
            expired = [k for k, v in state["audio_emb_cache"].items() if v["expires"] < now]
            for k in expired:
                del state["audio_emb_cache"][k]


# ── Tokenizer (must match build_search_index.py) ─────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text):
    if not text or not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]

def tokenize_and_hash(text, secret_key=LYRICS_HMAC_KEY):
    tokens = tokenize(text)
    return [hmac.new(secret_key, t.encode(), hashlib.sha256).hexdigest()[:16] for t in tokens]


# ── Startup / Shutdown ───────────────────────────────────────────────────────
def load_indices():
    """Load all search indices into memory."""
    log.info("Loading search indices...")

    # FAISS indices (including whisper)
    _skip_faiss = {s.strip() for s in os.environ.get("LAIONTUNES_SKIP_FAISS", "").split(",") if s.strip()}
    for field in ["tag", "lyric", "mood", "caption", "transcription", "whisper"]:
        if field in _skip_faiss:
            log.info(f"  FAISS {field}: skipped by LAIONTUNES_SKIP_FAISS")
            continue
        idx_path = INDEX_DIR / f"faiss_{field}.index"
        map_path = INDEX_DIR / f"idmap_{field}.npy"
        if idx_path.exists() and map_path.exists():
            state["faiss"][field] = faiss.read_index(str(idx_path))
            state["id_maps"][field] = np.load(str(map_path))
            log.info(f"  FAISS {field}: {state['faiss'][field].ntotal:,} vectors")
        else:
            log.info(f"  FAISS {field}: not found (skipped)")

    # Build reverse mapping for whisper (row_id -> faiss_idx)
    if "whisper" in state["id_maps"]:
        idmap = state["id_maps"]["whisper"]
        state["whisper_rid_to_idx"] = {int(rid): idx for idx, rid in enumerate(idmap)}
        log.info(f"  Whisper reverse map: {len(state['whisper_rid_to_idx']):,} entries")

    # BM25 indices
    _skip_bm25 = {s.strip() for s in os.environ.get("LAIONTUNES_SKIP_BM25", "").split(",") if s.strip()}
    for field in ["tags", "caption", "transcription", "lyrics_hashed"]:
        if field in _skip_bm25:
            log.info(f"  BM25 {field}: skipped by LAIONTUNES_SKIP_BM25")
            continue
        pkl_path = INDEX_DIR / f"bm25_{field}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                state["bm25"][field] = pickle.load(f)
            log.info(f"  BM25 {field}: {len(state['bm25'][field].row_ids):,} documents")
        else:
            log.info(f"  BM25 {field}: not found (skipped)")

    # SQLite
    state["db_path"] = str(INDEX_DIR / "metadata.db")
    conn = sqlite3.connect(state["db_path"])
    state["total_tracks"] = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    conn.close()
    log.info(f"  SQLite: {state['total_tracks']:,} tracks")

    # ── Instrumental subset indices ──────────────────────────────────
    inst = state["instrumental"]
    inst_db = INSTRUMENTAL_INDEX_DIR / "instrumental_metadata.db"
    if inst_db.exists():
        log.info("Loading instrumental subset indices...")
        for field in ["whisper", "caption"]:
            idx_path = INSTRUMENTAL_INDEX_DIR / f"faiss_{field}.index"
            map_path = INSTRUMENTAL_INDEX_DIR / f"idmap_{field}.npy"
            if idx_path.exists() and map_path.exists():
                inst["faiss"][field] = faiss.read_index(str(idx_path))
                inst["id_maps"][field] = np.load(str(map_path))
                log.info(f"  Instrumental FAISS {field}: {inst['faiss'][field].ntotal:,} vectors")
        if "whisper" in inst["id_maps"]:
            idmap = inst["id_maps"]["whisper"]
            inst["whisper_rid_to_idx"] = {int(rid): idx for idx, rid in enumerate(idmap)}
            log.info(f"  Instrumental whisper reverse map: {len(inst['whisper_rid_to_idx']):,} entries")
        for field in ["caption", "tags"]:
            pkl_path = INSTRUMENTAL_INDEX_DIR / f"bm25_{field}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    inst["bm25"][field] = pickle.load(f)
                log.info(f"  Instrumental BM25 {field}: {len(inst['bm25'][field].row_ids):,} documents")
        inst["db_path"] = str(inst_db)
        conn = sqlite3.connect(inst["db_path"])
        inst["total_tracks"] = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        conn.close()
        log.info(f"  Instrumental SQLite: {inst['total_tracks']:,} tracks")
    else:
        log.info("Instrumental subset indices not found (run filter_subset.py to build)")


class TEIEmbedder:
    """Wraps Hugging Face Text Embeddings Inference HTTP API to match SentenceTransformer interface."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        import requests as _req
        self._session = _req.Session()
        # Warm up / verify
        r = self._session.post(f"{self.url}/embed", json={"inputs": "test"})
        r.raise_for_status()
        self.dim = len(r.json()[0])
        log.info(f"  TEI connected at {self.url} (dim={self.dim})")

    def encode(self, text, normalize_embeddings=True):
        r = self._session.post(f"{self.url}/embed", json={"inputs": text})
        r.raise_for_status()
        emb = np.array(r.json()[0], dtype=np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        return emb


def load_embedder(gpu_id):
    """Load EmbeddingGemma 300M for real-time query embedding.
    Supports TEI backend (--embedder tei --tei-url URL) or SentenceTransformer.
    """
    tei_url = getattr(app.state, "tei_url", None)

    if tei_url:
        log.info(f"Connecting to TEI backend at {tei_url}...")
        state["embedder"] = TEIEmbedder(tei_url)
        return

    hf_cache = str(BASE_DIR / ".hf_cache")
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache

    import torch
    from sentence_transformers import SentenceTransformer

    # IMPORTANT: the parquet embeddings stored in FAISS were computed with
    # google/embeddinggemma-300m via SentenceTransformer, which uses
    # EmbeddingGemma's proper pooling (LastToken) and task prompts.
    # The onnx-community ONNX port does NOT ship a sentence-transformers
    # config, so it falls back to naive mean-pooling — that yields
    # a completely different embedding subspace (~0.04 cosine instead of ~1.0
    # for the same text). We MUST use the PyTorch model unless the user
    # explicitly opts into the (broken-for-this-index) ONNX path.
    use_onnx = os.environ.get("LAIONTUNES_USE_ONNX_EMBEDDER", "0") == "1"
    if not use_onnx:
        log.info(f"Loading EmbeddingGemma 300M (PyTorch) on GPU {gpu_id}...")
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            "google/embeddinggemma-300m",
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        state["embedder"] = model
        log.info(f"  EmbeddingGemma loaded on {device}")
        return

    # Opt-in ONNX path (faster on CPU but mismatched for the current FAISS indices)
    log.info("Loading EmbeddingGemma 300M (ONNX quantized, CPU) — opt-in")
    model = SentenceTransformer(
        "onnx-community/embeddinggemma-300m-ONNX",
        backend="onnx",
        model_kwargs={"file_name": "onnx/model_quantized.onnx"},
    )
    state["embedder"] = model
    log.warning(
        "  ONNX embedder loaded — note: retrieval quality will be degraded "
        "because the FAISS indices were built with PyTorch google/embeddinggemma-300m."
    )


def load_whisper_encoder(gpu_id):
    """Load laion/music-whisper encoder for audio similarity search."""
    if "whisper" not in state["faiss"]:
        log.info("  Whisper FAISS index not found, skipping encoder load")
        return

    cache_dir = str(BASE_DIR / ".hf_cache_embeddings")
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    import torch
    from transformers import WhisperProcessor, WhisperModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading Music-Whisper encoder on {device}...")
    processor = WhisperProcessor.from_pretrained("laion/music-whisper", cache_dir=cache_dir)
    model = WhisperModel.from_pretrained("laion/music-whisper", cache_dir=cache_dir)
    if device != "cpu":
        encoder = model.encoder.to(device).half().eval()
    else:
        encoder = model.encoder.eval()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state["whisper_processor"] = processor
    state["whisper_encoder"] = encoder
    state["whisper_device"] = device
    log.info(f"  Music-Whisper encoder loaded on {device}")

    # Restore HF_HOME for EmbeddingGemma
    hf_cache = str(BASE_DIR / ".hf_cache")
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_indices()
    load_embedder(gpu_id=app.state.gpu_id)
    if not getattr(app.state, "no_whisper", False):
        load_whisper_encoder(gpu_id=app.state.gpu_id)
    else:
        log.info("Whisper encoder loading skipped (--no-whisper)")
    yield
    # Shutdown (nothing to clean up)


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="LAION-Tunes Search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request/Response models ──────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    negative_query: Optional[str] = None    # Negative prompt for vector search (subtracted from query)
    search_type: str = "bm25"               # "vector" | "bm25" | "combined"
    vector_field: str = "caption"           # "tag" | "lyric" | "mood" | "caption" | "transcription"
    bm25_field: str = "caption"             # "tags" | "caption" | "transcription" | "lyrics_hashed"
    rank_by: str = "similarity"             # "similarity" | "aesthetics" | "plays" | "likes"
    min_aesthetics: Optional[float] = None  # Filter: minimum score_average
    min_similarity: Optional[float] = None  # Filter: minimum cosine similarity
    subset_filter: Optional[str] = None     # Filter: "suno" | "udio" etc
    vocal_filter: Optional[str] = None      # "instrumental" | "vocals" | None (all)
    min_duration: Optional[float] = 60.0    # Minimum duration in seconds (default 1 min)
    languages: Optional[list[str]] = None   # List of language codes to include, None = all
    negative_weight: float = 0.7            # Weight for negative query subtraction (0-1)
    nsfw_filter: Optional[str] = None       # None=all, "sfw_only", "nsfw_only"
    top_k: int = 50
    # Two-stage search
    stage2_enabled: bool = False
    stage2_query: Optional[str] = None
    stage2_search_type: str = "vector"      # "vector" | "bm25"
    stage2_vector_field: str = "caption"
    stage2_bm25_field: str = "caption"
    stage2_top_k: int = 50


class SimilarSearchRequest(BaseModel):
    row_id: int
    top_k: int = 50
    rank_by: str = "similarity"
    min_aesthetics: Optional[float] = None
    subset_filter: Optional[str] = None
    vocal_filter: Optional[str] = None
    min_duration: Optional[float] = 60.0
    languages: Optional[list[str]] = None
    nsfw_filter: Optional[str] = None
    # Stage 2
    stage2_enabled: bool = False
    stage2_query: Optional[str] = None
    stage2_search_type: str = "vector"
    stage2_vector_field: str = "caption"
    stage2_bm25_field: str = "caption"
    stage2_top_k: int = 50


# ── Helper: fetch metadata from SQLite ───────────────────────────────────────
def fetch_tracks(row_ids, conn):
    """Fetch track metadata for given row_ids from SQLite."""
    if len(row_ids) == 0:
        return {}
    placeholders = ",".join("?" * len(row_ids))
    cursor = conn.execute(
        f"SELECT * FROM tracks WHERE row_id IN ({placeholders})",
        list(int(r) for r in row_ids),
    )
    columns = [desc[0] for desc in cursor.description]
    result = {}
    for row in cursor:
        d = dict(zip(columns, row))
        result[d["row_id"]] = d
    return result


def format_result(track, score=None, score_type="similarity", has_whisper_emb=False):
    """Format a track dict for API response."""
    # Parse JSON array fields
    genre = json.loads(track.get("genre_tags") or "[]")
    scene = json.loads(track.get("scene_tags") or "[]")
    emotion = json.loads(track.get("emotion_tags") or "[]")

    return {
        "row_id": track["row_id"],
        "title": track.get("title") or "Untitled",
        "audio_url": track.get("audio_url") or "",
        "subset": track.get("subset") or "",
        "tags_text": track.get("tags_text") or "",
        "mood_text": track.get("mood_text") or "",
        "genre_tags": genre,
        "scene_tags": scene,
        "emotion_tags": emotion,
        "score_average": track.get("score_average"),
        "score_coherence": track.get("score_coherence"),
        "score_musicality": track.get("score_musicality"),
        "score_memorability": track.get("score_memorability"),
        "score_clarity": track.get("score_clarity"),
        "score_naturalness": track.get("score_naturalness"),
        "play_count": track.get("play_count") or 0,
        "upvote_count": track.get("upvote_count") or 0,
        "duration_seconds": track.get("duration_seconds"),
        "music_whisper_caption": track.get("music_whisper_caption") or "",
        "has_caption": bool(track.get("has_caption")),
        "has_transcription": bool(track.get("has_transcription")),
        "is_instrumental": bool(track.get("is_instrumental")),
        "language": track.get("language") or "unknown",
        "score": round(float(score), 4) if score is not None else None,
        "score_type": score_type,
        "has_whisper_emb": has_whisper_emb,
        # NSFW safety labels
        "nsfw_overall_label": track.get("nsfw_overall_label") or "likely_sfw",
        "nsfw_gore_label": track.get("nsfw_gore_label") or "likely_sfw",
        "nsfw_sexual_label": track.get("nsfw_sexual_label") or "likely_sfw",
        "nsfw_hate_label": track.get("nsfw_hate_label") or "likely_sfw",
        "nsfw_gore_sim": round(float(track["nsfw_gore_sim"]), 4) if track.get("nsfw_gore_sim") is not None else None,
        "nsfw_sexual_sim": round(float(track["nsfw_sexual_sim"]), 4) if track.get("nsfw_sexual_sim") is not None else None,
        "nsfw_hate_sim": round(float(track["nsfw_hate_sim"]), 4) if track.get("nsfw_hate_sim") is not None else None,
    }


# ── Search logic ─────────────────────────────────────────────────────────────
def _get_indices(subset_filter=None):
    """Return (faiss_dict, id_maps_dict, bm25_dict, db_path, total_tracks, whisper_rid_to_idx)
    for the given subset filter. Uses instrumental subset indices when applicable."""
    if subset_filter == "instrumental_subset" and state["instrumental"]["db_path"]:
        inst = state["instrumental"]
        return inst["faiss"], inst["id_maps"], inst["bm25"], inst["db_path"], inst["total_tracks"], inst["whisper_rid_to_idx"]
    return state["faiss"], state["id_maps"], state["bm25"], state["db_path"], state["total_tracks"], state["whisper_rid_to_idx"]


def vector_search(query_embedding, field, top_k, subset_filter=None):
    """Search FAISS index, return (row_ids, similarities)."""
    faiss_dict, id_maps_dict, _, _, _, _ = _get_indices(subset_filter)
    if field not in faiss_dict:
        return np.array([]), np.array([])

    index = faiss_dict[field]
    id_map = id_maps_dict[field]

    # Reshape for FAISS
    qvec = query_embedding.reshape(1, -1).astype(np.float32)
    k = min(top_k, index.ntotal)
    if k == 0:
        return np.array([]), np.array([])

    similarities, indices = index.search(qvec, k)
    similarities = similarities[0]
    indices = indices[0]

    # Map FAISS indices to row_ids
    valid = indices >= 0
    row_ids = id_map[indices[valid]]
    sims = similarities[valid]
    return row_ids, sims


def bm25_search(query_text, field, top_k, subset_filter=None):
    """BM25 text search, return (row_ids, scores)."""
    _, _, bm25_dict, _, _, _ = _get_indices(subset_filter)
    if field not in bm25_dict:
        return np.array([]), np.array([])

    # Hash tokens for lyrics search
    if field == "lyrics_hashed":
        tokens = tokenize_and_hash(query_text)
    else:
        tokens = tokenize(query_text)

    return bm25_dict[field].search(tokens, top_k=top_k)


def apply_filters(tracks_dict, min_aesthetics=None, subset_filter=None, min_similarity=None,
                   scores=None, vocal_filter=None, min_duration=None, languages=None,
                   nsfw_filter=None):
    """Filter tracks by criteria. Returns filtered row_ids.
    nsfw_filter: None=all, 'sfw_only'=exclude NSFW, 'nsfw_only'=only NSFW
    """
    filtered = []
    for row_id, track in tracks_dict.items():
        if min_aesthetics is not None:
            avg = track.get("score_average")
            if avg is None or avg < min_aesthetics:
                continue
        if subset_filter:
            if subset_filter == "no_riffusion":
                if track.get("subset") == "riffusion":
                    continue
            elif track.get("subset") != subset_filter:
                continue
        if min_similarity is not None and scores is not None:
            sim = scores.get(row_id, 0)
            if sim < min_similarity:
                continue
        if vocal_filter == "instrumental":
            if not track.get("is_instrumental"):
                continue
        elif vocal_filter == "vocals":
            if track.get("is_instrumental"):
                continue
        if min_duration is not None:
            dur = track.get("duration_seconds")
            if dur is not None and dur < min_duration:
                continue
        if languages:
            lang = track.get("language") or "unknown"
            if lang not in languages:
                continue
        if nsfw_filter == "sfw_only":
            label = track.get("nsfw_overall_label") or "likely_sfw"
            if label != "likely_sfw":
                continue
        elif nsfw_filter == "nsfw_only":
            label = track.get("nsfw_overall_label") or "likely_sfw"
            if label == "likely_sfw":
                continue
        filtered.append(row_id)
    return filtered


def rank_results(row_ids, tracks_dict, rank_by, sim_scores=None):
    """Re-rank row_ids by the specified field."""
    if rank_by in ("similarity", "music_similarity") and sim_scores:
        return sorted(row_ids, key=lambda r: sim_scores.get(r, 0), reverse=True)
    elif rank_by == "aesthetics":
        return sorted(row_ids, key=lambda r: tracks_dict.get(r, {}).get("score_average") or 0, reverse=True)
    elif rank_by == "plays":
        return sorted(row_ids, key=lambda r: tracks_dict.get(r, {}).get("play_count") or 0, reverse=True)
    elif rank_by == "likes":
        return sorted(row_ids, key=lambda r: tracks_dict.get(r, {}).get("upvote_count") or 0, reverse=True)
    return row_ids


# ── Audio embedding helper ──────────────────────────────────────────────────
def compute_audio_embedding(audio_bytes):
    """Compute mean-pooled Whisper encoder embedding from audio bytes.
    Returns L2-normalized 768-dim float32 numpy array.
    """
    import torch
    import librosa
    import soundfile as sf

    processor = state["whisper_processor"]
    encoder = state["whisper_encoder"]
    device = state["whisper_device"]

    if processor is None or encoder is None:
        raise RuntimeError("Whisper encoder not loaded")

    # Write bytes to temp file, load with librosa
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            audio, sr = librosa.load(tmp.name, sr=16000, mono=True)
        except Exception:
            # Try soundfile as fallback
            audio, sr = sf.read(tmp.name)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Trim to first 30 seconds
    max_samples = 30 * 16000
    audio = audio[:max_samples].astype(np.float32)

    if len(audio) < 1600:  # less than 0.1s
        raise ValueError("Audio too short (< 0.1 seconds)")

    # Process through Whisper encoder
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    if device != "cpu":
        input_features = input_features.half()

    with torch.no_grad():
        outputs = encoder(input_features)
        hidden = outputs.last_hidden_state  # (1, seq, 768)
        mean_pooled = hidden.mean(dim=1)    # (1, 768)
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        embedding = mean_pooled.cpu().float().numpy()[0]  # (768,)

    return embedding


def search_by_whisper_embedding(query_embedding, top_k, rank_by, min_aesthetics,
                                  subset_filter, vocal_filter, min_duration, languages,
                                  nsfw_filter=None):
    """Search whisper FAISS index with a query embedding, apply filters, return formatted results."""
    t0 = time.time()

    use_subset = subset_filter == "instrumental_subset"
    _, _, _, db_path, total_tracks, whisper_rid_set = _get_indices(subset_filter)
    effective_subset = None if use_subset else subset_filter

    has_filters = (effective_subset or vocal_filter or languages
                   or (min_aesthetics and min_aesthetics > 0)
                   or (min_duration and min_duration > 0))
    fetch_k = max(top_k * 100, 20000) if has_filters else max(top_k * 10, 2000)

    # Search FAISS whisper index
    row_ids, sims = vector_search(query_embedding, "whisper", fetch_k,
                                   subset_filter=subset_filter)
    sim_scores = {}
    candidate_row_ids = set()
    for rid, sim in zip(row_ids, sims):
        rid = int(rid)
        sim_scores[rid] = float(sim)
        candidate_row_ids.add(rid)

    # Fetch metadata
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tracks_dict = fetch_tracks(list(candidate_row_ids), conn)

    # Apply filters
    filtered_ids = apply_filters(
        tracks_dict,
        min_aesthetics=min_aesthetics,
        subset_filter=effective_subset,
        vocal_filter=None if use_subset else vocal_filter,
        min_duration=min_duration,
        languages=languages,
        nsfw_filter=nsfw_filter,
    )

    # Rank
    ranked_ids = rank_results(filtered_ids, tracks_dict, rank_by, sim_scores)
    final_ids = ranked_ids[:top_k]

    # Build response
    results = []
    for rid in final_ids:
        track = tracks_dict.get(rid)
        if not track:
            continue
        score = sim_scores.get(rid, 0)
        score_type = "cosine_similarity"
        if rank_by == "aesthetics":
            score = track.get("score_average") or 0
            score_type = "aesthetics"
        elif rank_by == "plays":
            score = track.get("play_count") or 0
            score_type = "play_count"
        elif rank_by == "likes":
            score = track.get("upvote_count") or 0
            score_type = "upvote_count"

        r = format_result(dict(track), score=score, score_type=score_type,
                          has_whisper_emb=(rid in whisper_rid_set))
        results.append(r)

    conn.close()
    search_time_ms = (time.time() - t0) * 1000

    return {
        "results": results,
        "total_candidates": len(candidate_row_ids),
        "total_filtered": len(filtered_ids),
        "total_tracks": total_tracks,
        "search_time_ms": round(search_time_ms, 1),
        "search_type": "music_similarity",
        "vector_field": "whisper",
    }


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the search frontend."""
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/nsfw-report", response_class=HTMLResponse)
async def serve_nsfw_report():
    """Serve the NSFW safety analysis report."""
    report = BASE_DIR / "nsfw_safety_report.html"
    if report.exists():
        return report.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Report not generated yet</h1>", status_code=404)


@app.post("/api/search")
async def search(req: SearchRequest):
    """Main search endpoint supporting vector, BM25, and combined search."""
    t0 = time.time()

    # Embed query (and optional negative query)
    t_emb_start = time.time()
    query_embedding = state["embedder"].encode(
        req.query, normalize_embeddings=True
    ).astype(np.float32)

    # Negative prompt: subtract negative embedding from positive, then re-normalize
    if req.negative_query and req.negative_query.strip() and req.search_type in ("vector", "combined"):
        neg_embedding = state["embedder"].encode(
            req.negative_query.strip(), normalize_embeddings=True
        ).astype(np.float32)
        weight = max(0.0, min(1.0, req.negative_weight))
        query_embedding = query_embedding - weight * neg_embedding
        # Re-normalize to unit length for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

    emb_time_ms = (time.time() - t_emb_start) * 1000

    # Determine which indices to use (instrumental subset or main)
    use_subset = req.subset_filter == "instrumental_subset"
    _, _, _, db_path, total_tracks, whisper_rid_set = _get_indices(req.subset_filter)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Determine candidate pool size - fetch large pool so filters don't starve results.
    # FAISS IndexFlatIP scans all vectors regardless of k, so large k is ~free.
    effective_subset = None if use_subset else req.subset_filter
    has_filters = (effective_subset or req.vocal_filter or req.languages
                   or (req.min_aesthetics and req.min_aesthetics > 0)
                   or (req.min_duration and req.min_duration > 0))
    fetch_k = max(req.top_k * 100, 20000) if has_filters else max(req.top_k * 10, 2000)

    sim_scores = {}  # row_id -> similarity score
    bm25_scores = {}  # row_id -> bm25 score
    candidate_row_ids = set()

    if req.search_type in ("vector", "combined"):
        row_ids, sims = vector_search(query_embedding, req.vector_field, fetch_k,
                                       subset_filter=req.subset_filter)
        for rid, sim in zip(row_ids, sims):
            rid = int(rid)
            sim_scores[rid] = float(sim)
            candidate_row_ids.add(rid)

    if req.search_type in ("bm25", "combined"):
        row_ids, scores = bm25_search(req.query, req.bm25_field, fetch_k,
                                       subset_filter=req.subset_filter)
        for rid, score in zip(row_ids, scores):
            rid = int(rid)
            bm25_scores[rid] = float(score)
            candidate_row_ids.add(rid)

    if req.search_type == "combined" and sim_scores and bm25_scores:
        # For combined: intersect vector and BM25 candidates
        candidate_row_ids = set(sim_scores.keys()) & set(bm25_scores.keys())
        if not candidate_row_ids:
            # If no intersection, fall back to union
            candidate_row_ids = set(sim_scores.keys()) | set(bm25_scores.keys())

    # Fetch metadata for candidates
    tracks_dict = fetch_tracks(list(candidate_row_ids), conn)

    # Use primary scores based on search type for similarity filtering
    primary_scores = sim_scores if req.search_type in ("vector", "combined") else bm25_scores

    # Apply filters (skip subset/vocal filters when using instrumental subset)
    filtered_ids = apply_filters(
        tracks_dict,
        min_aesthetics=req.min_aesthetics,
        subset_filter=effective_subset,
        min_similarity=req.min_similarity,
        scores=primary_scores,
        vocal_filter=None if use_subset else req.vocal_filter,
        min_duration=req.min_duration,
        languages=req.languages,
        nsfw_filter=req.nsfw_filter,
    )

    # Rank
    ranked_ids = rank_results(filtered_ids, tracks_dict, req.rank_by, primary_scores)

    # Take top_k
    stage1_ids = ranked_ids[:req.top_k]

    # ── Stage 2: re-filter by a second query ────────────────────────────
    stage2_scores = {}
    stage2_info = None
    if req.stage2_enabled and req.stage2_query and req.stage2_query.strip():
        stage1_set = set(stage1_ids)
        s2_query = req.stage2_query.strip()

        if req.stage2_search_type == "vector":
            s2_emb = state["embedder"].encode(s2_query, normalize_embeddings=True).astype(np.float32)
            # Search with large K to cover stage1 candidates
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_sims = vector_search(s2_emb, req.stage2_vector_field, s2_fetch_k,
                                              subset_filter=req.subset_filter)
            for rid, sim in zip(s2_rids, s2_sims):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sim)
        else:  # bm25
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_scores_arr = bm25_search(s2_query, req.stage2_bm25_field, s2_fetch_k,
                                                   subset_filter=req.subset_filter)
            for rid, sc in zip(s2_rids, s2_scores_arr):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sc)

        # Rank stage1 candidates by stage2 score, take stage2_top_k
        stage1_ids = sorted(stage1_ids, key=lambda r: stage2_scores.get(r, -999), reverse=True)
        stage1_ids = stage1_ids[:req.stage2_top_k]
        stage2_info = {
            "query": s2_query,
            "search_type": req.stage2_search_type,
            "field": req.stage2_vector_field if req.stage2_search_type == "vector" else req.stage2_bm25_field,
            "matched": len(stage2_scores),
            "returned": len(stage1_ids),
        }

    final_ids = stage1_ids

    # Build response
    results = []
    for rid in final_ids:
        track = tracks_dict.get(rid)
        if not track:
            continue
        # Pick the best score to show
        if stage2_scores:
            score = stage2_scores.get(rid, 0)
            score_type = "cosine_similarity" if req.stage2_search_type == "vector" else "bm25"
        else:
            score = primary_scores.get(rid, 0)
            score_type = "cosine_similarity" if req.search_type == "vector" else "bm25"
        if req.rank_by == "aesthetics" and not stage2_scores:
            score = track.get("score_average") or 0
            score_type = "aesthetics"
        elif req.rank_by == "plays" and not stage2_scores:
            score = track.get("play_count") or 0
            score_type = "play_count"
        elif req.rank_by == "likes" and not stage2_scores:
            score = track.get("upvote_count") or 0
            score_type = "upvote_count"

        r = format_result(dict(track), score=score, score_type=score_type,
                          has_whisper_emb=(rid in whisper_rid_set))
        # Attach both scores when stage2 is active
        if stage2_scores:
            r["stage1_score"] = round(primary_scores.get(rid, 0), 4)
            r["stage2_score"] = round(stage2_scores.get(rid, 0), 4)
        results.append(r)

    conn.close()
    search_time_ms = (time.time() - t0) * 1000

    resp = {
        "results": results,
        "total_candidates": len(candidate_row_ids),
        "total_filtered": len(filtered_ids),
        "total_tracks": total_tracks,
        "search_time_ms": round(search_time_ms, 1),
        "query_embedding_time_ms": round(emb_time_ms, 1),
        "search_type": req.search_type,
        "vector_field": req.vector_field,
        "bm25_field": req.bm25_field,
    }
    if stage2_info:
        resp["stage2"] = stage2_info
    return resp


@app.post("/api/search_by_audio")
async def search_by_audio(
    request: Request,
    audio: UploadFile = File(...),
    top_k: int = Form(50),
    rank_by: str = Form("similarity"),
    subset_filter: Optional[str] = Form(None),
    vocal_filter: Optional[str] = Form(None),
    min_duration: Optional[float] = Form(None),
    min_aesthetics: Optional[float] = Form(None),
    languages: Optional[str] = Form(None),  # comma-separated language codes
    nsfw_filter: Optional[str] = Form(None),  # None=all, "sfw_only", "nsfw_only"
    # Stage 2 fields
    stage2_enabled: Optional[str] = Form(None),
    stage2_query: Optional[str] = Form(None),
    stage2_search_type: Optional[str] = Form("vector"),
    stage2_vector_field: Optional[str] = Form("caption"),
    stage2_bm25_field: Optional[str] = Form("caption"),
    stage2_top_k: int = Form(50),
):
    """Search by uploaded audio: compute whisper embedding, find nearest neighbors."""
    if "whisper" not in state["faiss"]:
        return JSONResponse(status_code=503, content={"error": "Whisper index not available"})
    if state["whisper_encoder"] is None:
        return JSONResponse(status_code=503, content={"error": "Whisper encoder not loaded"})

    t0 = time.time()

    # Client IP for caching
    client_ip = request.client.host if request.client else "unknown"
    filename = audio.filename or "unknown"

    # Check cache
    cached_emb = cache_get(client_ip, filename)
    cache_hit = cached_emb is not None

    if cache_hit:
        query_embedding = cached_emb
        emb_time_ms = 0.0
        log.info(f"Audio embedding cache hit: {filename} from {client_ip}")
    else:
        # Read audio bytes
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        if len(audio_bytes) > 100 * 1024 * 1024:  # 100 MB limit
            return JSONResponse(status_code=400, content={"error": "File too large (max 100 MB)"})

        # Compute embedding
        t_emb = time.time()
        try:
            query_embedding = compute_audio_embedding(audio_bytes)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Audio processing failed: {str(e)}"})
        emb_time_ms = (time.time() - t_emb) * 1000

        # Cache the result
        cache_set(client_ip, filename, query_embedding)
        log.info(f"Cached audio embedding: {filename} from {client_ip} ({emb_time_ms:.0f}ms)")

    # Parse languages
    lang_list = None
    if languages and languages.strip():
        lang_list = [l.strip() for l in languages.split(",") if l.strip()]

    # Parse optional floats that come as "None" strings from form
    min_dur = min_duration if min_duration and min_duration > 0 else None
    min_aes = min_aesthetics if min_aesthetics and min_aesthetics > 0 else None
    sub_filter = subset_filter if subset_filter and subset_filter != "null" else None
    voc_filter = vocal_filter if vocal_filter and vocal_filter != "null" else None

    nsfw_f = nsfw_filter if nsfw_filter and nsfw_filter not in ("null", "") else None

    result = search_by_whisper_embedding(
        query_embedding, top_k, rank_by, min_aes, sub_filter, voc_filter, min_dur, lang_list,
        nsfw_filter=nsfw_f,
    )
    result["query_embedding_time_ms"] = round(emb_time_ms, 1)
    result["audio_filename"] = filename
    result["cache_hit"] = cache_hit

    # Stage 2: re-filter music similarity results by text query
    s2_enabled = stage2_enabled and stage2_enabled.lower() in ("true", "1", "yes")
    if s2_enabled and stage2_query and stage2_query.strip():
        stage1_results = result["results"]
        stage1_set = {r["row_id"] for r in stage1_results}
        s2_query = stage2_query.strip()
        stage2_scores = {}

        if stage2_search_type == "vector":
            s2_emb = state["embedder"].encode(s2_query, normalize_embeddings=True).astype(np.float32)
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_sims = vector_search(s2_emb, stage2_vector_field, s2_fetch_k,
                                              subset_filter=sub_filter)
            for rid, sim in zip(s2_rids, s2_sims):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sim)
        else:  # bm25
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_scores_arr = bm25_search(s2_query, stage2_bm25_field, s2_fetch_k,
                                                   subset_filter=sub_filter)
            for rid, sc in zip(s2_rids, s2_scores_arr):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sc)

        # Re-rank by stage2 score
        for r in stage1_results:
            r["stage1_score"] = r.get("score", 0)
            r["stage2_score"] = round(stage2_scores.get(r["row_id"], -999), 4)

        stage1_results.sort(key=lambda r: r["stage2_score"], reverse=True)
        result["results"] = stage1_results[:stage2_top_k]
        result["stage2"] = {
            "query": s2_query,
            "search_type": stage2_search_type,
            "field": stage2_vector_field if stage2_search_type == "vector" else stage2_bm25_field,
            "matched": len(stage2_scores),
            "returned": len(result["results"]),
        }

    return result


@app.post("/api/search_similar")
async def search_similar(req: SimilarSearchRequest):
    """Search for tracks similar to a given sample using pre-computed whisper embeddings."""
    faiss_dict, _, _, db_path, _, whisper_rid_to_idx = _get_indices(req.subset_filter)

    if "whisper" not in faiss_dict:
        return JSONResponse(status_code=503, content={"error": "Whisper index not available"})

    # Try subset first, then fall back to main index for the reference embedding
    rid_to_idx = whisper_rid_to_idx
    whisper_faiss = faiss_dict["whisper"]
    if req.row_id not in rid_to_idx:
        # Fall back to main index for the reference embedding lookup
        if req.row_id in state["whisper_rid_to_idx"]:
            rid_to_idx = state["whisper_rid_to_idx"]
            whisper_faiss = state["faiss"]["whisper"]
        else:
            return JSONResponse(status_code=404, content={
                "error": f"No whisper embedding found for row_id {req.row_id}"
            })

    t0 = time.time()

    # Reconstruct the embedding from FAISS index
    faiss_idx = rid_to_idx[req.row_id]
    query_embedding = whisper_faiss.reconstruct(faiss_idx)

    result = search_by_whisper_embedding(
        query_embedding, req.top_k, req.rank_by,
        req.min_aesthetics, req.subset_filter, req.vocal_filter,
        req.min_duration, req.languages,
        nsfw_filter=req.nsfw_filter,
    )
    result["reference_row_id"] = req.row_id

    # Look up title for the reference
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT title FROM tracks WHERE row_id=?", (req.row_id,)).fetchone()
    if not row:
        # Try main db
        conn.close()
        conn = sqlite3.connect(state["db_path"])
        row = conn.execute("SELECT title FROM tracks WHERE row_id=?", (req.row_id,)).fetchone()
    conn.close()
    if row:
        result["reference_title"] = row[0] or "Untitled"

    # Stage 2: re-filter music similarity results by text query
    if req.stage2_enabled and req.stage2_query and req.stage2_query.strip():
        stage1_results = result["results"]
        stage1_set = {r["row_id"] for r in stage1_results}
        s2_query = req.stage2_query.strip()
        stage2_scores = {}

        if req.stage2_search_type == "vector":
            s2_emb = state["embedder"].encode(s2_query, normalize_embeddings=True).astype(np.float32)
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_sims = vector_search(s2_emb, req.stage2_vector_field, s2_fetch_k,
                                              subset_filter=req.subset_filter)
            for rid, sim in zip(s2_rids, s2_sims):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sim)
        else:  # bm25
            s2_fetch_k = max(len(stage1_set) * 20, 50000)
            s2_rids, s2_scores_arr = bm25_search(s2_query, req.stage2_bm25_field, s2_fetch_k,
                                                   subset_filter=req.subset_filter)
            for rid, sc in zip(s2_rids, s2_scores_arr):
                rid = int(rid)
                if rid in stage1_set:
                    stage2_scores[rid] = float(sc)

        for r in stage1_results:
            r["stage1_score"] = r.get("score", 0)
            r["stage2_score"] = round(stage2_scores.get(r["row_id"], -999), 4)

        stage1_results.sort(key=lambda r: r["stage2_score"], reverse=True)
        result["results"] = stage1_results[:req.stage2_top_k]
        result["stage2"] = {
            "query": s2_query,
            "search_type": req.stage2_search_type,
            "field": req.stage2_vector_field if req.stage2_search_type == "vector" else req.stage2_bm25_field,
            "matched": len(stage2_scores),
            "returned": len(result["results"]),
        }

    return result


@app.get("/api/stats")
async def stats():
    """Dataset statistics."""
    conn = sqlite3.connect(state["db_path"])
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    subsets = {}
    for row in conn.execute("SELECT subset, COUNT(*) FROM tracks GROUP BY subset"):
        subsets[row[0]] = row[1]

    avg_scores = {}
    for row in conn.execute(
        "SELECT AVG(score_average), MIN(score_average), MAX(score_average) FROM tracks WHERE score_average IS NOT NULL"
    ):
        avg_scores = {"mean": round(row[0], 3) if row[0] else None, "min": row[1], "max": row[2]}

    with_caption = conn.execute("SELECT COUNT(*) FROM tracks WHERE has_caption=1").fetchone()[0]
    with_transcription = conn.execute("SELECT COUNT(*) FROM tracks WHERE has_transcription=1").fetchone()[0]

    faiss_stats = {}
    for field, idx in state["faiss"].items():
        faiss_stats[field] = idx.ntotal

    bm25_stats = {}
    for field, idx in state["bm25"].items():
        bm25_stats[field] = len(idx.row_ids)

    # Language distribution
    languages = {}
    for row in conn.execute(
        "SELECT language, COUNT(*) FROM tracks WHERE language IS NOT NULL AND language != '' "
        "GROUP BY language ORDER BY COUNT(*) DESC"
    ):
        languages[row[0]] = row[1]

    instrumental = conn.execute("SELECT COUNT(*) FROM tracks WHERE is_instrumental=1").fetchone()[0]

    conn.close()

    return {
        "total_tracks": total,
        "subsets": subsets,
        "score_average": avg_scores,
        "with_caption": with_caption,
        "with_transcription": with_transcription,
        "faiss_indices": faiss_stats,
        "bm25_indices": bm25_stats,
        "languages": languages,
        "instrumental_count": instrumental,
        "whisper_embeddings": state["faiss"]["whisper"].ntotal if "whisper" in state["faiss"] else 0,
        "instrumental_subset_tracks": state["instrumental"]["total_tracks"],
    }


class TakedownRequest(BaseModel):
    track_info: str = ""
    reason: str
    statement: str
    name: str
    email: str
    phone: str = ""


TAKEDOWN_LOG = Path(__file__).parent / "logs" / "takedown_requests.jsonl"

@app.post("/api/takedown")
async def takedown(req: TakedownRequest, request: Request):
    """Record a rights-holder takedown / removal request."""
    import uuid as _uuid
    from datetime import datetime as _dt

    if not req.reason or not req.statement or not req.name or not req.email:
        return JSONResponse(status_code=422, content={"detail": "Missing required fields."})
    if "@" not in req.email:
        return JSONResponse(status_code=422, content={"detail": "Invalid e-mail address."})

    ref_id = "TD-" + _uuid.uuid4().hex[:10].upper()
    record = {
        "ref_id": ref_id,
        "timestamp": _dt.utcnow().isoformat() + "Z",
        "ip": request.client.host if request.client else "unknown",
        "track_info": req.track_info,
        "reason": req.reason,
        "statement": req.statement,
        "name": req.name,
        "email": req.email,
        "phone": req.phone,
    }
    try:
        TAKEDOWN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TAKEDOWN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("Failed to write takedown log: %s", e)
        return JSONResponse(status_code=500, content={"detail": "Could not record request."})

    logging.info("Takedown request %s from %s <%s> — %s", ref_id, req.name, req.email, req.reason)
    return {"ref_id": ref_id}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LAION-Tunes Search Server")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for EmbeddingGemma")
    parser.add_argument("--tei-url", type=str, default=None,
                        help="Use TEI backend for embeddings (e.g. http://localhost:8090)")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Skip loading whisper encoder (for CPU-only deployment)")
    args = parser.parse_args()

    app.state.gpu_id = args.gpu
    app.state.tei_url = args.tei_url
    app.state.no_whisper = args.no_whisper

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
