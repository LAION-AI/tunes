"""Step 1 — Cache per-clip handcrafted features to disk.

For each clip (all splits), extracts:

  mel_stats    — log-mel statistics (mean + std per bin, 256-dim)
  mel_full     — full log-mel spectrogram (128, T) for LCNN training
  phase        — phase-coherence / group-delay statistics (20-dim)
  rolloff      — high-band energy ratios + spectral rolloff (8-dim)
  bicoherence  — simplified harmonic cross-correlation features (15-dim)
  chroma_ssm   — long-range chroma self-similarity statistics (6-dim)
  denoiser     — spectral smoother residual statistics (24-dim)
  stereo       — L/R coherence features (16-dim; 0-padded for mono sources)

Saved as NPZ per clip under:
  {CACHE_DIR}/features/{uuid}.npz

Usage:
  uv run scripts/training/cache_features.py [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import (
    CACHE_DIR,
    CLIP_SAMPLES,
    SAMPLE_RATE,
    build_song_manifest,
    load_audio_mono,
    load_audio_stereo,
    load_split,
)

FEATURE_DIR = CACHE_DIR / "features"
N_FFT = 1024
HOP_LENGTH = 160           # 10 ms at 16 kHz
N_MELS = 128
N_CQT_BINS = 84            # 7 octaves × 12 bins
F_MIN = 32.7               # C1

# ─── Per-probe extractors ─────────────────────────────────────────────────────

def _log_mel(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mel_stats 256-dim, mel_full 128×T)."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=60.0, fmax=8000.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    stats = np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])
    return stats.astype(np.float32), log_mel.astype(np.float32)


def _phase_coherence(wav: np.ndarray) -> np.ndarray:
    """Group-delay deviation per sub-band → 20-dim."""
    D = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.unwrap(np.angle(D), axis=1)   # unwrap along time
    # Group delay ≈ derivative of phase w.r.t. frequency (along freq axis)
    gd = np.diff(phase, axis=0)              # (N_FFT/2, T)
    # Split into 10 sub-bands, compute std of gd per band per frame, then mean+std
    n_bands = 10
    band_size = gd.shape[0] // n_bands
    feats = []
    for i in range(n_bands):
        band = gd[i * band_size : (i + 1) * band_size, :]
        gd_std_per_frame = band.std(axis=0)
        feats.extend([float(gd_std_per_frame.mean()), float(gd_std_per_frame.std())])
    return np.array(feats, dtype=np.float32)


def _high_band_rolloff(wav: np.ndarray) -> np.ndarray:
    """Energy above 16/18/20 kHz + spectral rolloff → 8-dim."""
    # Full FFT magnitude
    fft_mag = np.abs(np.fft.rfft(wav, n=N_FFT * 8))
    freqs = np.fft.rfftfreq(N_FFT * 8, d=1.0 / SAMPLE_RATE)
    total_e = float(np.sum(fft_mag ** 2)) + 1e-9
    thresholds = [4000, 8000, 12000, 16000, 18000, 20000]
    feats = []
    for t in thresholds:
        e = float(np.sum(fft_mag[freqs >= t] ** 2))
        feats.append(e / total_e)
    # Spectral rolloff at 85% and 95%
    cum = np.cumsum(fft_mag ** 2)
    cum /= cum[-1] + 1e-9
    feats.append(float(freqs[min(np.searchsorted(cum, 0.85), len(freqs) - 1)]) / (SAMPLE_RATE / 2))
    feats.append(float(freqs[min(np.searchsorted(cum, 0.95), len(freqs) - 1)]) / (SAMPLE_RATE / 2))
    return np.array(feats, dtype=np.float32)


def _bicoherence(wav: np.ndarray) -> np.ndarray:
    """Simplified: harmonic cross-correlation via CQT energy ratios → 15-dim.

    Measures whether harmonics are coupled as expected from real instruments.
    """
    C = np.abs(librosa.cqt(
        wav, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        n_bins=N_CQT_BINS, fmin=F_MIN,
    ))
    # Energy per semitone bin, averaged over time
    e_bin = C.mean(axis=1)   # (84,)
    # For each pitch bin, compute correlation with its 1st, 2nd, 3rd, 4th harmonic
    feats = []
    for harmonic in [12, 19, 24, 28, 31]:   # semitone offsets for harmonics 2–6
        valid = e_bin.shape[0] - harmonic
        if valid <= 0:
            feats.append(0.0)
            continue
        base = e_bin[:valid]
        harm = e_bin[harmonic : harmonic + valid]
        corr = float(np.corrcoef(base, harm)[0, 1])
        feats.append(0.0 if np.isnan(corr) else corr)
    # CQT spectral flux between adjacent frames (mean+std+skew per octave)
    flux = np.diff(C, axis=1) ** 2
    n_oct = N_CQT_BINS // 12
    for oct_i in range(n_oct):
        band = flux[oct_i * 12 : (oct_i + 1) * 12, :]
        feats.append(float(band.mean()))
    # Trim/pad to 15-dim
    feats = feats[:15]
    while len(feats) < 15:
        feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def _chroma_ssm(wav: np.ndarray) -> np.ndarray:
    """Long-range chroma self-similarity → 6-dim.

    Autoregressive token models lose sectional coherence; this captures that.
    """
    chroma = librosa.feature.chroma_cqt(
        y=wav, sr=SAMPLE_RATE, hop_length=HOP_LENGTH * 10, bins_per_octave=36,
    )                                        # (12, ~300) for 30 s
    # Downsample to 1 frame/s ≈ 30 frames
    chroma_ds = chroma[:, ::max(1, chroma.shape[1] // 30)][:, :30]
    # Cosine similarity matrix
    norms = np.linalg.norm(chroma_ds, axis=0, keepdims=True) + 1e-9
    chroma_n = chroma_ds / norms
    ssm = chroma_n.T @ chroma_n          # (30, 30)
    upper = ssm[np.triu_indices(len(ssm), k=1)]
    return np.array([
        float(upper.mean()),
        float(upper.std()),
        float(np.percentile(upper, 25)),
        float(np.percentile(upper, 75)),
        float((upper > 0.9).mean()),      # fraction of very-similar pairs
        float((upper < 0.3).mean()),      # fraction of very-dissimilar pairs
    ], dtype=np.float32)


def _denoiser_residual(wav: np.ndarray) -> np.ndarray:
    """Spectral over-smoothing residual → 24-dim.

    Proxy for 'how much does a simple denoiser remove'. Generative artifacts
    tend to be at specific frequencies and show up as non-Gaussian residuals.
    """
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=96,
    )
    log_mel = librosa.power_to_db(mel)
    # Smoothed: median filter across time
    from scipy.ndimage import median_filter
    smoothed = median_filter(log_mel, size=(1, 15))  # 1.5 s window
    residual = log_mel - smoothed
    # Statistics per 8 frequency bands
    n_bands = 8
    band_size = residual.shape[0] // n_bands
    feats = []
    for i in range(n_bands):
        band = residual[i * band_size : (i + 1) * band_size, :]
        feats.extend([
            float(band.mean()),
            float(band.std()),
            float(np.abs(band).mean()),
        ])
    return np.array(feats, dtype=np.float32)


def _stereo_coherence(stereo_wav: np.ndarray) -> np.ndarray:
    """L/R phase + amplitude coherence → 16-dim.

    Many generators produce mono or weakly correlated pseudo-stereo.
    """
    if stereo_wav.shape[0] < 2:
        return np.zeros(16, dtype=np.float32)
    L, R = stereo_wav[0], stereo_wav[1]
    DL = librosa.stft(L, n_fft=N_FFT, hop_length=HOP_LENGTH)
    DR = librosa.stft(R, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag_L, phase_L = np.abs(DL), np.angle(DL)
    mag_R, phase_R = np.abs(DR), np.angle(DR)
    phase_diff = np.abs(phase_L - phase_R)
    # Magnitude correlation per sub-band
    n_bands = 8
    freq_size = DL.shape[0] // n_bands
    feats = []
    for i in range(n_bands):
        sl = slice(i * freq_size, (i + 1) * freq_size)
        r = float(np.corrcoef(
            mag_L[sl].ravel(), mag_R[sl].ravel()
        )[0, 1])
        feats.append(0.0 if np.isnan(r) else r)
        pd_band = phase_diff[sl].ravel()
        feats.append(float(pd_band.mean()))
    return np.array(feats, dtype=np.float32)


# ─── Per-clip extraction ───────────────────────────────────────────────────────

def extract_all(entry: dict) -> dict[str, np.ndarray] | None:
    path = entry["_audio_path"]
    try:
        wav_mono = load_audio_mono(path)
        wav_stereo = load_audio_stereo(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] failed to load {path}: {exc}", flush=True)
        return None

    mel_stats, mel_full = _log_mel(wav_mono)
    return {
        "mel_stats":    mel_stats,
        "mel_full":     mel_full,
        "phase":        _phase_coherence(wav_mono),
        "rolloff":      _high_band_rolloff(wav_mono),
        "bicoherence":  _bicoherence(wav_mono),
        "chroma_ssm":   _chroma_ssm(wav_mono),
        "denoiser":     _denoiser_residual(wav_mono),
        "stereo":       _stereo_coherence(wav_stereo),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (keep 1 on MPS — CPU feature extraction)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap per split (for quick smoke-tests)")
    args = parser.parse_args()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_song_manifest()

    all_records: list[dict] = []
    for split in ("training", "validation", "test", "test-ood"):
        recs = load_split(split, manifest, max_records=args.limit)
        for r in recs:
            r["_split"] = split
        all_records.extend(recs)

    # Deduplicate by uuid (some entries appear in multiple splits)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_records:
        if r["uuid"] not in seen:
            seen.add(r["uuid"])
            unique.append(r)

    print(f"Extracting features for {len(unique)} unique clips → {FEATURE_DIR}")

    cached = skipped = errors = 0
    for entry in tqdm(unique, ncols=90):
        out_path = FEATURE_DIR / f"{entry['uuid']}.npz"
        if out_path.exists():
            cached += 1
            continue
        feats = extract_all(entry)
        if feats is None:
            errors += 1
            continue
        np.savez(out_path, **feats)
        skipped = skipped  # keep variable for summary

    print(f"Done. cached={cached}  extracted={len(unique)-cached-errors}  errors={errors}")


if __name__ == "__main__":
    main()
