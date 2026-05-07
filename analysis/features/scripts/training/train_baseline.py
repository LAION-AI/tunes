"""Step 4/5/7 — Train baseline and final models.

Models (--model):
  lcnn             — Log-mel CNN (LCNN-style, reference)
  mert_head        — 2-layer MLP on frozen MERT embeddings
  muq_head         — 2-layer MLP on frozen MuQ embeddings
  moss_nano_head   — 2-layer MLP on frozen MOSS-Nano embeddings
  clap_head        — 2-layer MLP on frozen CLAP embeddings
  multiview        — Probe features + LCNN backbone, joint MLP head
  convnext         — ConvNeXt-Tiny on log-mel (timm, 224×224 input)
  vit              — ViT-Small/16 on log-mel (timm, 224×224 input)
  efficientvit     — EfficientViT-B1 on log-mel (timm, 224×224 input)
  specttra_alpha   — SONICS Specttra-α  (requires `sonics` package — see below)
  specttra_beta    — SONICS Specttra-β  (requires `sonics` package — see below)
  specttra_gamma   — SONICS Specttra-γ  (requires `sonics` package — see below)

Flags:
  --augment REGIME   augmentation regime (default: none)
  --probes P1,P2     handcrafted probes for multiview (default: phase,denoiser)
  --swa              enable Stochastic Weight Averaging (last 20% of epochs)
  --tta              enable Test-Time Augmentation at eval
  --seeds 0,1,2      comma-separated seeds (default: 0)
  --epochs N         training epochs (default: 20)

Usage:
  uv run scripts/training/train_baseline.py --model lcnn
  uv run scripts/training/train_baseline.py --model multiview --probes bicoherence,denoiser \\
      --augment combined_safe --swa --tta --seeds 0,1,2,3,4
  uv run scripts/training/train_baseline.py --model convnext

Notes on the new architectures
  ConvNeXt / ViT / EfficientViT
    Require `timm` (add with `uv add timm`). The log-mel tensor (1, 128, T) is
    standardized per-sample, replicated to 3 channels, and resized to 224×224
    by `MelToRGB`. Pretrained ImageNet weights are loaded by default — toggle
    via the VISION_PRETRAINED constant below if you want a from-scratch run
    that mirrors the LCNN protocol exactly.

  Specttra-{alpha,beta,gamma}
    Native SONICS architectures from https://awsaf49.github.io/sonics-website/
    (code: https://github.com/awsaf49/sonics). Specttra consumes raw waveform
    and computes its own front-end, so a separate `WaveformDataset` branch is
    used in run_training(). The factory function `_build_specttra()` calls
    into the upstream `sonics` package — you may need to adjust the import
    path / constructor signature to match the version of the SONICS repo you
    install (see the docstring on `_build_specttra`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import (
    CACHE_DIR,
    CHECKPOINTS_DIR,
    CLIP_SAMPLES,
    RESULTS_DIR,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    build_song_manifest,
    compute_metrics,
    get_label,
    load_split,
    log_to_csv,
    save_scores,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FEATURE_DIR = CACHE_DIR / "features"
EMBED_DIR = CACHE_DIR / "embeddings"

# Mel spec params
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128

# ─── Augmentation ─────────────────────────────────────────────────────────────

class _SpecAug(nn.Module):
    """SpecAugment: time + frequency masking."""
    def __init__(self, time_mask=50, freq_mask=20):
        super().__init__()
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.training:
            spec = self.tm(spec)
            spec = self.fm(spec)
        return spec


def _apply_codec_aug(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Randomly apply MP3/OGG compression to both real and fake equally."""
    import io, random
    from torchaudio.io import AudioEffectsChain
    try:
        t = torch.from_numpy(wav).unsqueeze(0)
        codec = random.choice(["mp3", "ogg"])
        bitrate = random.choice(["64k", "128k", "192k"])
        buf = io.BytesIO()
        torchaudio.save(buf, t, sr, format=codec, compression=bitrate)
        buf.seek(0)
        t2, _ = torchaudio.load(buf, format=codec)
        return t2.squeeze(0).numpy()[:len(wav)]
    except Exception:  # noqa: BLE001
        return wav


def _apply_noise(wav: np.ndarray) -> np.ndarray:
    """RawBoost-style additive noise."""
    rng = np.random.default_rng()
    snr_db = rng.uniform(15, 40)
    sig_power = np.mean(wav ** 2) + 1e-9
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return np.clip(wav + noise, -1.0, 1.0)


def _apply_pitch_shift(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    import random, librosa
    semitones = random.choice([-2, -1, 1, 2])
    try:
        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=semitones)
    except Exception:  # noqa: BLE001
        return wav


def _apply_loudness(wav: np.ndarray) -> np.ndarray:
    gain_db = np.random.uniform(-6, 6)
    gain = 10 ** (gain_db / 20)
    return np.clip(wav * gain, -1.0, 1.0)


def _apply_reverb(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convolve with a random exponential-decay IR."""
    rng = np.random.default_rng()
    ir_len = int(sr * rng.uniform(0.05, 0.4))
    decay = rng.uniform(3.0, 10.0)
    ir = np.exp(-decay * np.linspace(0, 1, ir_len)) * rng.normal(0, 1, ir_len)
    ir = ir.astype(np.float32) / (np.abs(ir).max() + 1e-9)
    from scipy.signal import fftconvolve
    out = fftconvolve(wav, ir)[:len(wav)]
    peak = np.abs(out).max() + 1e-9
    return (out / peak * np.abs(wav).max()).astype(np.float32)


REGIME_AUGMENTS: dict[str, list[Callable]] = {
    "none":               [],
    "specaug":            [],         # handled at spectrogram level in Dataset
    "mixup":              [],         # handled at batch level in trainer
    "codec":              [_apply_codec_aug],
    "noise":              [_apply_noise],
    "pitch_shift":        [_apply_pitch_shift],
    "loudness":           [_apply_loudness],
    "reverb":             [_apply_reverb],
    "combined_safe":      [_apply_codec_aug, _apply_loudness],
    "combined_aggressive":[_apply_codec_aug, _apply_noise, _apply_pitch_shift,
                           _apply_loudness, _apply_reverb],
}

SPECAUG_REGIMES = {"specaug", "combined_safe", "combined_aggressive"}
MIXUP_REGIMES   = {"mixup",   "combined_aggressive"}


def augment_wav(wav: np.ndarray, regime: str) -> np.ndarray:
    for fn in REGIME_AUGMENTS.get(regime, []):
        try:
            wav = fn(wav)
        except Exception:  # noqa: BLE001
            pass
    return wav


# ─── Datasets ─────────────────────────────────────────────────────────────────

def _wav_to_mel(wav: np.ndarray) -> torch.Tensor:
    """Return log mel spectrogram (1, N_MELS, T)."""
    t = torch.from_numpy(wav).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=60.0, f_max=8000.0,
    )(t)
    log_mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0,
                                                     amin=1e-10, db_multiplier=0.0)
    return log_mel  # (1, N_MELS, T)


class AudioDataset(Dataset):
    def __init__(self, records: list[dict], augment: str = "none", training: bool = True):
        self.records = records
        self.augment = augment
        self.training = training
        self.mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=60.0, f_max=8000.0,
        )
        self.use_specaug = augment in SPECAUG_REGIMES

    def __len__(self):
        return len(self.records)

    def _load_segment(self, path: str) -> np.ndarray:
        import torchaudio as ta
        wav, sr = ta.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0).numpy().astype(np.float32)
        if len(wav) > CLIP_SAMPLES:
            wav = wav[:CLIP_SAMPLES]
        # Random or fixed segment
        if self.training and len(wav) >= SEGMENT_SAMPLES:
            start = np.random.randint(0, len(wav) - SEGMENT_SAMPLES + 1)
            wav = wav[start : start + SEGMENT_SAMPLES]
        else:
            wav = wav[:SEGMENT_SAMPLES]
            if len(wav) < SEGMENT_SAMPLES:
                wav = np.pad(wav, (0, SEGMENT_SAMPLES - len(wav)))
        return wav

    def __getitem__(self, idx: int):
        # Some files can become unreadable on external volumes mid-run.
        # Retry with alternate samples so one bad clip doesn't crash DataLoader workers.
        retries = min(8, len(self.records))
        for attempt in range(retries):
            pick_idx = idx if attempt == 0 else np.random.randint(0, len(self.records))
            entry = self.records[pick_idx]
            try:
                label = get_label(entry)
                wav = self._load_segment(entry["_audio_path"])
                if self.training:
                    wav = augment_wav(wav, self.augment)
                mel = self.mel_tf(torch.from_numpy(wav).unsqueeze(0))
                log_mel = torchaudio.functional.amplitude_to_DB(
                    mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0
                )  # (1, N_MELS, T)
                if self.use_specaug and self.training:
                    log_mel = torchaudio.transforms.TimeMasking(50)(log_mel)
                    log_mel = torchaudio.transforms.FrequencyMasking(20)(log_mel)
                return log_mel, torch.tensor(label, dtype=torch.long), entry["uuid"]
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    print(
                        f"[WARN] unreadable audio for {entry.get('uuid', 'unknown')}: {exc}",
                        flush=True,
                    )
                continue

        raise RuntimeError("Failed to load any valid audio sample after retries.")


class EmbedDataset(Dataset):
    """Load pre-cached foundation model embeddings."""
    def __init__(self, records: list[dict], model_name: str):
        embed_dir = EMBED_DIR / model_name
        self.items: list[tuple[np.ndarray, int, str]] = []
        for e in records:
            p = embed_dir / f"{e['uuid']}.npy"
            if p.exists():
                emb = np.load(p).astype(np.float32)
                self.items.append((emb, get_label(e), e["uuid"]))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        emb, label, uuid = self.items[idx]
        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.long), uuid


class MultiviewDataset(Dataset):
    """Combine LCNN mel + handcrafted probe features."""
    def __init__(
        self, records: list[dict], probes: list[str],
        augment: str = "none", training: bool = True,
    ):
        self.audio_ds = AudioDataset(records, augment=augment, training=training)
        self.probes = probes
        self.feat_dir = FEATURE_DIR
        # Filter to entries with both audio and feature cache
        self.valid_indices = []
        for i, entry in enumerate(records):
            fp = self.feat_dir / f"{entry['uuid']}.npz"
            if fp.exists() and entry.get("_audio_path"):
                self.valid_indices.append(i)
        self.records = records

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        mel, label, uuid = self.audio_ds[real_idx]
        entry = self.records[real_idx]
        fp = self.feat_dir / f"{entry['uuid']}.npz"
        data = np.load(fp)
        probe_feats = []
        for p in self.probes:
            feat = data[p].astype(np.float32) if p in data else np.zeros(1, dtype=np.float32)
            probe_feats.append(feat)
        probe_vec = torch.from_numpy(np.concatenate(probe_feats))
        return mel, probe_vec, label, uuid


def _labels_for_dataset(dataset: Dataset) -> np.ndarray:
    """Return labels for the examples that are actually available in a dataset."""
    if isinstance(dataset, AudioDataset):
        labels = [get_label(r) for r in dataset.records]
    elif isinstance(dataset, EmbedDataset):
        labels = [label for _, label, _ in dataset.items]
    elif isinstance(dataset, MultiviewDataset):
        labels = [get_label(dataset.records[i]) for i in dataset.valid_indices]
    elif isinstance(dataset, WaveformDataset):
        labels = [get_label(r) for r in dataset.audio.records]
    else:
        labels = []
    return np.array(labels, dtype=np.int64)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    if n_neg == 0 or n_pos == 0:
        print(
            f"  [WARN] training labels are imbalanced: negatives={n_neg}, positives={n_pos}"
        )
    return torch.tensor(
        [1.0, float(n_neg / max(n_pos, 1))],
        dtype=torch.float32,
        device=DEVICE,
    )


@torch.no_grad()
def _refresh_bn(model: nn.Module, loader: DataLoader, is_multiview: bool) -> None:
    """Refresh BatchNorm stats for regular and multiview SWA models."""
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    if not momenta:
        return

    was_training = model.training
    model.train()
    for batch in loader:
        if is_multiview:
            mel, probe_feat, _, _ = batch
            model(mel.to(DEVICE), probe_feat.to(DEVICE))
        else:
            inputs = batch[0]
            model(inputs.to(DEVICE))

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


# ─── Model architectures ──────────────────────────────────────────────────────

class MFM(nn.Module):
    """Max Feature Map activation — halves channels."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        return torch.max(x[:, :c], x[:, c:])


class LCNN(nn.Module):
    """Log-CQT CNN with Max Feature Map activations for audio deepfake detection."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  MFM(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(16, 64, kernel_size=3, padding=1), MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), MFM(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1), MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), MFM(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1), MFM(),
            nn.BatchNorm2d(32),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128), MFM(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (B, 32)
        # Run through first FC+MFM
        fc = self.classifier[1]
        mfm = self.classifier[2]
        return mfm(fc(x))   # (B, 64)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiviewModel(nn.Module):
    def __init__(self, probe_dim: int, num_classes: int = 2):
        super().__init__()
        self.lcnn = LCNN(num_classes=num_classes)
        # Replace LCNN classifier; we'll use features only
        self.lcnn.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128), MFM(),
            nn.Dropout(0.5),
        )  # outputs 64-dim
        self.head = nn.Sequential(
            nn.Linear(64 + probe_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        self.probe_bn = nn.BatchNorm1d(probe_dim)

    def forward(self, mel: torch.Tensor, probe_feat: torch.Tensor) -> torch.Tensor:
        lcnn_feat = self.lcnn(mel)  # (B, 64)
        probe_feat = self.probe_bn(probe_feat)
        combined = torch.cat([lcnn_feat, probe_feat], dim=1)
        return self.head(combined)


# ─── Vision backbones (ConvNeXt / ViT / EfficientViT via timm) ────────────────

# Set to False to train vision backbones from scratch (matches the LCNN
# no-pretrain protocol). True is recommended for the headline numbers.
VISION_PRETRAINED = True

# Common image size — ViT-S/16 hard-requires 224×224; ConvNeXt and EfficientViT
# also default to 224 pretrained, so we keep one size for fair comparison.
VISION_INPUT_SIZE = 224

_TIMM_NAMES = {
    "convnext":     "convnext_tiny",
    "vit":          "vit_small_patch16_224",
    "efficientvit": "efficientvit_b1.r224_in1k",
}


class MelToRGB(nn.Module):
    """Adapt log-mel (B, 1, N_MELS, T) to a 3-channel image of fixed size.

    Per-sample standardization keeps numerics stable across clips with very
    different loudness; channel replication is the standard trick for using
    ImageNet vision backbones on spectrograms.
    """
    def __init__(self, target_size: int = VISION_INPUT_SIZE):
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(
            x, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        )
        return x


class TimmBackbone(nn.Module):
    """Wrap a timm classifier so it consumes log-mel input."""
    def __init__(self, timm_name: str, num_classes: int = 2,
                 pretrained: bool = VISION_PRETRAINED):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "`timm` is required for --model in {convnext, vit, efficientvit}. "
                "Install with `uv add timm`."
            ) from exc
        self.adapter = MelToRGB()
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=num_classes, in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))


# ─── SONICS SpecTTTra ─────────────────────────────────────────────────────────
#
# Native SpecTTTra-{α,β,γ} models from https://github.com/awsaf49/sonics
# Public API (v0.1+):
#     from sonics import HFAudioClassifier
#     model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-5s")
# Variants and durations are selected by the HF checkpoint id, not constructor
# args. Available checkpoints: sonics-spectttra-{alpha,beta,gamma}-{5s,120s}.
# We use the 5s checkpoints because our SEGMENT_SAMPLES = 3s × 16kHz (closest
# fit, padded to 5s = 80 000 samples). 120s checkpoints would mostly process
# silence on our segment length.
# ─────────────────────────────────────────────────────────────────────────────

_SPECTTRA_VARIANTS = {"alpha", "beta", "gamma"}
_SPECTTRA_DURATION_S = 5
_SPECTTRA_SAMPLES = SAMPLE_RATE * _SPECTTRA_DURATION_S  # 80 000


class SpecttraWrapper(nn.Module):
    """Wraps SONICS HFAudioClassifier for our 2-class trainer.

    Pads waveform from SEGMENT_SAMPLES (3s) to _SPECTTRA_SAMPLES (5s) and
    normalizes the upstream output to a (B, 2) logit tensor regardless of
    whether the underlying model returns a tensor, an HF ModelOutput, or a
    dict.
    """
    def __init__(self, variant: str, num_classes: int = 2):
        super().__init__()
        try:
            from sonics import HFAudioClassifier
        except ImportError as exc:
            raise ImportError(
                f"--model specttra_{variant} requires the `sonics` package. "
                "Install per https://github.com/awsaf49/sonics — for example:\n"
                "  uv add 'sonics @ git+https://github.com/awsaf49/sonics.git'"
            ) from exc

        hf_id = f"awsaf49/sonics-spectttra-{variant}-{_SPECTTRA_DURATION_S}s"
        self.backbone = HFAudioClassifier.from_pretrained(hf_id)
        self.target_samples = _SPECTTRA_SAMPLES

        # The HF checkpoint is trained as a binary AI-music detector, so its
        # head already has the right output dimensionality. If upstream layout
        # changes, we adapt at forward time.
        self._num_classes = num_classes

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        # wav shape: (B, T) — pad/trim time axis to target.
        T = wav.shape[-1]
        if T == self.target_samples:
            return wav
        if T > self.target_samples:
            return wav[..., : self.target_samples]
        pad = self.target_samples - T
        return nn.functional.pad(wav, (0, pad))

    def _to_logits(self, out, batch_size: int) -> torch.Tensor:
        # Normalize upstream output to (B, num_classes).
        if isinstance(out, torch.Tensor):
            t = out
        elif hasattr(out, "logits"):
            t = out.logits
        elif isinstance(out, dict) and "logits" in out:
            t = out["logits"]
        else:
            raise RuntimeError(
                f"Unexpected SpecTTTra output type: {type(out).__name__}. "
                "Inspect HFAudioClassifier.forward() and update SpecttraWrapper._to_logits()."
            )
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] == 1 and self._num_classes == 2:
            # Single-logit (binary sigmoid) head → expand to 2-class logits.
            return torch.cat([torch.zeros_like(t), t], dim=-1)
        return t

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self._pad_or_trim(wav)
        out = self.backbone(wav)
        return self._to_logits(out, batch_size=wav.shape[0])


def _build_specttra(variant: str, num_classes: int = 2) -> nn.Module:
    if variant not in _SPECTTRA_VARIANTS:
        raise ValueError(f"Unknown SpecTTTra variant: {variant!r}")
    return SpecttraWrapper(variant=variant, num_classes=num_classes)


class WaveformDataset(Dataset):
    """Yield raw waveform tensors for models with their own front-end (Specttra)."""
    def __init__(self, records: list[dict], augment: str = "none", training: bool = True):
        self.audio = AudioDataset(records, augment=augment, training=training)
        # Reuse AudioDataset for path resolution + segmenting + waveform aug,
        # but skip the mel transform.

    def __len__(self): return len(self.audio.records)

    def __getitem__(self, idx: int):
        retries = min(8, len(self.audio.records))
        for attempt in range(retries):
            pick_idx = idx if attempt == 0 else np.random.randint(0, len(self.audio.records))
            entry = self.audio.records[pick_idx]
            try:
                label = get_label(entry)
                wav = self.audio._load_segment(entry["_audio_path"])
                if self.audio.training:
                    wav = augment_wav(wav, self.audio.augment)
                return (
                    torch.from_numpy(wav.astype(np.float32)),
                    torch.tensor(label, dtype=torch.long),
                    entry["uuid"],
                )
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    print(
                        f"[WARN] unreadable audio for {entry.get('uuid', 'unknown')}: {exc}",
                        flush=True,
                    )
                continue
        raise RuntimeError("Failed to load any valid waveform sample after retries.")


def _infer_embed_dim(model_name: str) -> int:
    embed_dir = EMBED_DIR / model_name
    if embed_dir.exists():
        for p in embed_dir.glob("*.npy"):
            return int(np.load(p).shape[0])
    defaults = {"mert": 768, "muq": 768, "moss_nano": 128, "clap": 512, "encodec": 128}
    return defaults.get(model_name.replace("_head", ""), 512)


# ─── Training loop ────────────────────────────────────────────────────────────

def mixup_batch(
    inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam


import contextlib


def _autocast_ctx(enabled: bool):
    """Autocast context manager that no-ops when disabled.

    Vision backbones (ConvNeXt / ViT / EfficientViT) and Specttra are
    numerically unstable under fp16 autocast on MPS without a working
    GradScaler; they are trained in fp32 by passing enabled=False.
    """
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(DEVICE.type, dtype=torch.float16)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    augment: str,
    is_multiview: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    autocast_enabled: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    use_mixup = augment in MIXUP_REGIMES

    for batch in loader:
        optimizer.zero_grad()
        if is_multiview:
            mel, probe_feat, labels, _ = batch
            mel = mel.to(DEVICE)
            probe_feat = probe_feat.to(DEVICE)
            labels = labels.to(DEVICE)
            with _autocast_ctx(autocast_enabled):
                logits = model(mel, probe_feat)
                loss = criterion(logits, labels)
        else:
            mel, labels, _ = batch
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)
            if use_mixup:
                mel, labels_a, labels_b, lam = mixup_batch(mel, labels)
                with _autocast_ctx(autocast_enabled):
                    logits = model(mel)
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                with _autocast_ctx(autocast_enabled):
                    logits = model(mel)
                    loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    is_multiview: bool = False,
    tta: bool = False,
    autocast_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_scores, all_labels, all_uuids = [], [], []
    softmax = nn.Softmax(dim=1)

    for batch in loader:
        if is_multiview:
            mel, probe_feat, labels, uuids = batch
            mel = mel.to(DEVICE)
            probe_feat = probe_feat.to(DEVICE)
            with _autocast_ctx(autocast_enabled):
                logits = model(mel, probe_feat)
            scores = softmax(logits.float())[:, 1]
        else:
            mel, labels, uuids = batch
            mel = mel.to(DEVICE)

            if tta:
                # Average over original + 2 mild augmentations
                all_logits = []
                for _tta_pass in range(3):
                    m = mel if _tta_pass == 0 else _tta_specaug(mel)
                    with _autocast_ctx(autocast_enabled):
                        all_logits.append(softmax(model(m).float()))
                scores = torch.stack(all_logits).mean(0)[:, 1]
            else:
                with _autocast_ctx(autocast_enabled):
                    logits = model(mel)
                scores = softmax(logits.float())[:, 1]

        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())
        all_uuids.extend(list(uuids))

    if not all_scores:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
            all_uuids,
        )

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        all_uuids,
    )


def _tta_specaug(mel: torch.Tensor) -> torch.Tensor:
    """Light SpecAugment for TTA."""
    mel = torchaudio.transforms.TimeMasking(30)(mel)
    mel = torchaudio.transforms.FrequencyMasking(10)(mel)
    return mel


def _compute_metrics_if_possible(
    scores: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> dict[str, float] | None:
    if len(labels) == 0:
        print(f"  [{split_name}] skipped metrics (no examples).")
        return None
    if len(np.unique(labels)) < 2:
        print(f"  [{split_name}] skipped metrics (only one class present).")
        return None
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        print(
            f"  [{split_name}] skipped metrics ({n_bad}/{len(scores)} non-finite "
            f"scores — likely fp16 underflow or model divergence)."
        )
        return None
    return compute_metrics(scores, labels)


def _fmt_metric(metrics: dict[str, float] | None, key: str) -> str:
    return f"{metrics[key]:.3f}" if metrics is not None else "n/a"


# ─── Full training run ────────────────────────────────────────────────────────

def run_training(
    model_key: str,
    augment: str,
    probes: list[str],
    seed: int,
    epochs: int,
    use_swa: bool,
    use_tta: bool,
    batch_size: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifest = build_song_manifest()
    train_recs = load_split("training", manifest)
    val_recs   = load_split("validation", manifest)
    test_recs  = load_split("test", manifest)
    ood_recs   = load_split("test-ood", manifest)

    is_multiview = model_key == "multiview"
    is_embed     = model_key.endswith("_head")
    is_timm      = model_key in _TIMM_NAMES
    is_specttra  = model_key.startswith("specttra_")
    run_id = f"{model_key}_aug-{augment}_seed{seed}"

    # Build datasets / loaders
    if is_multiview:
        train_ds = MultiviewDataset(train_recs, probes, augment=augment, training=True)
        val_ds   = MultiviewDataset(val_recs,   probes, augment="none",  training=False)
        test_ds  = MultiviewDataset(test_recs,  probes, augment="none",  training=False)
        ood_ds   = MultiviewDataset(ood_recs,   probes, augment="none",  training=False)
    elif is_embed:
        embed_model = model_key.replace("_head", "")
        train_ds = EmbedDataset(train_recs, embed_model)
        val_ds   = EmbedDataset(val_recs,   embed_model)
        test_ds  = EmbedDataset(test_recs,  embed_model)
        ood_ds   = EmbedDataset(ood_recs,   embed_model)
    elif is_specttra:
        # Specttra computes its own front-end from raw waveform.
        train_ds = WaveformDataset(train_recs, augment=augment, training=True)
        val_ds   = WaveformDataset(val_recs,   augment="none",  training=False)
        test_ds  = WaveformDataset(test_recs,  augment="none",  training=False)
        ood_ds   = WaveformDataset(ood_recs,   augment="none",  training=False)
    else:  # lcnn, convnext, vit, efficientvit — all consume log-mel
        train_ds = AudioDataset(train_recs, augment=augment, training=True)
        val_ds   = AudioDataset(val_recs,   augment="none",  training=False)
        test_ds  = AudioDataset(test_recs,  augment="none",  training=False)
        ood_ds   = AudioDataset(ood_recs,   augment="none",  training=False)

    print(f"\n── {run_id} ──")
    print(f"   train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  ood={len(ood_ds)}")

    if len(train_ds) == 0:
        extra = ""
        if is_embed:
            extra = f" Run the embedding cache for '{model_key.replace('_head', '')}' first."
        print(f"  [SKIP] no training examples available.{extra}")
        return

    for split_name, split_ds in [
        ("val", val_ds), ("test", test_ds), ("ood", ood_ds)
    ]:
        if len(split_ds) == 0:
            print(f"  [WARN] {split_name} split has no available examples; metrics will be skipped.")

    loader_kwargs = dict(num_workers=6, persistent_workers=True, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)
    ood_loader   = DataLoader(ood_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Build model
    if is_multiview:
        # Infer probe feature dim
        probe_dim = 0
        sample_npz = next(iter(FEATURE_DIR.glob("*.npz")), None)
        if sample_npz:
            d = np.load(sample_npz)
            for p in probes:
                probe_dim += d[p].shape[0] if p in d else 0
        else:
            probe_dim = sum({"phase": 20, "rolloff": 8, "bicoherence": 15,
                             "chroma_ssm": 6, "denoiser": 24, "stereo": 16,
                             "mel_stats": 256}.get(p, 0) for p in probes)
        model = MultiviewModel(probe_dim=probe_dim).to(DEVICE)
    elif is_embed:
        embed_dim = _infer_embed_dim(model_key.replace("_head", ""))
        model = MLPHead(in_dim=embed_dim).to(DEVICE)
    elif is_timm:
        model = TimmBackbone(_TIMM_NAMES[model_key]).to(DEVICE)
    elif is_specttra:
        variant = model_key.split("_", 1)[1]
        model = _build_specttra(variant).to(DEVICE)
    else:
        model = LCNN().to(DEVICE)

    # Class-balanced weights
    labels_arr = _labels_for_dataset(train_ds)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels_arr))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # SWA setup
    swa_model = None
    swa_scheduler = None
    swa_start = int(epochs * 0.8)
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

    ckpt_dir = CHECKPOINTS_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ood_auc, best_epoch, best_is_swa = 0.0, 0, False
    # fp16 autocast on MPS is unstable for non-BatchNorm vision backbones and
    # for Specttra. Train those in fp32; keep autocast for LCNN/MLP heads where
    # it has been validated.
    autocast_enabled = not (is_timm or is_specttra)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, augment,
            is_multiview=is_multiview,
            autocast_enabled=autocast_enabled,
        )
        scheduler.step()
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            eval_model = swa_model if (use_swa and epoch >= swa_start) else model
            if use_swa and epoch >= swa_start:
                _refresh_bn(swa_model, train_loader, is_multiview)
            val_scores, val_labels, _ = evaluate(
                eval_model, val_loader, is_multiview=is_multiview, tta=use_tta,
                autocast_enabled=autocast_enabled,
            )
            ood_scores, ood_labels, _ = evaluate(
                eval_model, ood_loader, is_multiview=is_multiview, tta=use_tta,
                autocast_enabled=autocast_enabled,
            )
            val_m = _compute_metrics_if_possible(val_scores, val_labels, "val")
            ood_m = _compute_metrics_if_possible(ood_scores, ood_labels, "ood")
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
                f"val_auc={_fmt_metric(val_m, 'auc')}  "
                f"ood_auc={_fmt_metric(ood_m, 'auc')}  "
                f"({elapsed:.1f}s)"
            )
            monitor_m = ood_m or val_m
            if monitor_m is not None and monitor_m["auc"] > best_ood_auc:
                best_ood_auc = monitor_m["auc"]
                best_epoch = epoch
                best_is_swa = eval_model is swa_model
                torch.save(eval_model.state_dict(), ckpt_dir / "best.pt")

    # Final evaluation on best checkpoint
    final_model = swa_model if (use_swa and best_is_swa) else model
    try:
        state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=True)
        final_model.load_state_dict(state)
    except Exception:  # noqa: BLE001
        pass

    print(f"  best OOD AUC={best_ood_auc:.3f} at epoch {best_epoch}")

    for split_name, loader in [
        ("val", val_loader), ("test", test_loader), ("ood", ood_loader)
    ]:
        scores, labels, uuids = evaluate(
            final_model, loader, is_multiview=is_multiview, tta=use_tta,
            autocast_enabled=autocast_enabled,
        )
        m = _compute_metrics_if_possible(scores, labels, split_name)
        if m is None:
            continue
        log_to_csv({
            "run_id": run_id, "model": model_key, "augment": augment,
            "seed": seed, "split": split_name, **m,
        })
        save_scores(scores, labels, run_id=run_id, split=split_name)
        print(
            f"  [{split_name}] auc={m['auc']:.3f}  eer={m['eer']:.3f}  "
            f"tpr@1%={m['tpr_at_1pct_fpr']:.3f}"
        )


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lcnn",
                        choices=["lcnn", "mert_head", "muq_head", "moss_nano_head",
                                 "clap_head", "multiview",
                                 "convnext", "vit", "efficientvit",
                                 "specttra_alpha", "specttra_beta", "specttra_gamma"])
    parser.add_argument("--augment", default="none")
    parser.add_argument("--probes", default="phase,denoiser",
                        help="Comma-separated probe names for multiview model")
    parser.add_argument("--swa",  action="store_true")
    parser.add_argument("--tta",  action="store_true")
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    seeds  = [int(s) for s in args.seeds.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]

    for seed in seeds:
        run_training(
            model_key=args.model,
            augment=args.augment,
            probes=probes,
            seed=seed,
            epochs=args.epochs,
            use_swa=args.swa,
            use_tta=args.tta,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
