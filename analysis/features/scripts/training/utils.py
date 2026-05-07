"""Shared utilities: data loading, audio I/O, metrics, CSV logging."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Feature/embedding cache on NVME; fallback to local
_NVME = Path("/Volumes/NVME/songrating-experiments")
CACHE_DIR: Path = _NVME if _NVME.exists() else REPO_ROOT / ".feature-cache"

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE * 30          # 30-second clips
SEGMENT_SAMPLES = SAMPLE_RATE * 3        # 3-second segments for models

# ─── Manifest ─────────────────────────────────────────────────────────────────

def build_song_manifest() -> dict[str, str]:
    """Return song_id → local_audio_path from all known manifests."""
    manifest: dict[str, str] = {}

    # NVME full dataset
    for path in [
        Path("/Volumes/NVME/songrating-data/manifest.jsonl"),
        REPO_ROOT / "data-cache" / "manifest.jsonl",
    ]:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                e = json.loads(line)
                if e.get("status") in ("cached", "downloaded") and e.get("path"):
                    manifest[e["song_id"]] = e["path"]
    return manifest


def load_split(
    split_name: str, manifest: dict[str, str], max_records: int | None = None
) -> list[dict]:
    """Load split JSONL, resolve audio paths; skip entries missing audio."""
    path = DATA_DIR / f"{split_name}.jsonl"
    records, missing = [], 0
    with path.open() as f:
        for line in f:
            e = json.loads(line)
            ap = manifest.get(e["song_id"])
            if ap and Path(ap).exists():
                e["_audio_path"] = ap
                records.append(e)
            else:
                missing += 1
            if max_records and len(records) >= max_records:
                break
    if missing:
        print(f"[{split_name}] {missing} entries skipped (no local audio).")
    return records


def get_label(entry: dict) -> int:
    """0 = human, 1 = AI."""
    return 0 if entry.get("source", entry.get("label", "")) == "human" else 1


# ─── Audio loading ─────────────────────────────────────────────────────────────

def load_audio_mono(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load → mono → resample → 30 s clip. Returns float32 (CLIP_SAMPLES,)."""
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[1] > CLIP_SAMPLES:
        wav = wav[:, :CLIP_SAMPLES]
    elif wav.shape[1] < CLIP_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, CLIP_SAMPLES - wav.shape[1]))
    return wav.squeeze(0).numpy().astype(np.float32)


def load_audio_stereo(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load → stereo (or mono) → resample → 30 s. Returns float32 (C, CLIP_SAMPLES)."""
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[1] > CLIP_SAMPLES:
        wav = wav[:, :CLIP_SAMPLES]
    elif wav.shape[1] < CLIP_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, CLIP_SAMPLES - wav.shape[1]))
    return wav.numpy().astype(np.float32)


def get_segments(wav: np.ndarray, seg_samples: int = SEGMENT_SAMPLES) -> list[np.ndarray]:
    """Split a clip into non-overlapping segments of seg_samples."""
    segs = []
    for start in range(0, len(wav) - seg_samples + 1, seg_samples):
        segs.append(wav[start : start + seg_samples])
    return segs if segs else [wav[:seg_samples]]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """AUC, EER, TPR@1%FPR, TPR@0.1%FPR."""
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))

    def _tpr_at(target: float) -> float:
        idx = int(np.searchsorted(fpr, target, side="right")) - 1
        return float(tpr[max(0, min(idx, len(tpr) - 1))])

    return {
        "auc": auc,
        "eer": eer,
        "tpr_at_1pct_fpr": _tpr_at(0.01),
        "tpr_at_01pct_fpr": _tpr_at(0.001),
    }


# ─── CSV logging ──────────────────────────────────────────────────────────────

_COLS = [
    "run_id", "model", "augment", "seed", "split",
    "auc", "eer", "tpr_at_1pct_fpr", "tpr_at_01pct_fpr",
]


def log_to_csv(row: dict[str, Any], path: Path = SUMMARY_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_scores(scores: np.ndarray, labels: np.ndarray, run_id: str, split: str) -> None:
    """Persist per-clip scores for later significance testing."""
    out = RESULTS_DIR / "scores" / f"{run_id}_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, scores=scores, labels=labels)
