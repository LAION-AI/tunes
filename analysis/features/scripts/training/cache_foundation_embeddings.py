"""Step 2 — Cache foundation model embeddings to disk (one-time).

Supported models (--models flag, comma-separated):
  mert        — m-a-p/MERT-v1-95M  (music)
  muq         — OpenMuQ/MuQ-MuLan-large  (music)
  moss_nano   — OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano, encoder pre-quantization
  clap        — laion/larger_clap_music  (audio-text contrastive)
  encodec     — facebook/encodec_24khz  (neural codec encoder)

Embeddings saved as:
  {CACHE_DIR}/embeddings/{model}/{uuid}.npy   float16, (D,) averaged pool

Usage:
  uv run scripts/training/cache_foundation_embeddings.py --models mert,muq,moss_nano,clap
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import (
    CACHE_DIR,
    SAMPLE_RATE,
    build_song_manifest,
    load_audio_mono,
    load_split,
)

EMBED_DIR = CACHE_DIR / "embeddings"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── Model loaders ────────────────────────────────────────────────────────────

def _load_mert():
    from transformers import AutoModel, AutoProcessor
    name = "m-a-p/MERT-v1-95M"
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).half().to(DEVICE).eval()
    sr_model = 24_000  # MERT uses 24 kHz

    def embed(wav16: np.ndarray) -> np.ndarray:
        import torchaudio
        wav24 = torchaudio.functional.resample(
            torch.from_numpy(wav16).unsqueeze(0), SAMPLE_RATE, sr_model
        ).squeeze(0).numpy()
        inputs = proc(wav24, sampling_rate=sr_model, return_tensors="pt")
        inputs = {k: v.half().to(DEVICE) if v.dtype == torch.float32 else v.to(DEVICE)
                  for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast("mps" if DEVICE.type == "mps" else "cpu",
                                              dtype=torch.float16):
            out = model(**inputs, output_hidden_states=True)
        # Average last 4 hidden layers then pool over time
        hidden = torch.stack(out.hidden_states[-4:]).mean(0)  # (1, T, D)
        return hidden.squeeze(0).mean(0).float().cpu().numpy().astype(np.float16)

    return embed, "mert"


def _load_muq():
    from muq import MuQMuLan
    name = "OpenMuQ/MuQ-MuLan-large"
    model = MuQMuLan.from_pretrained(name).to(DEVICE).eval()
    sr_model = 24_000  # MuQ and MuQ-MuLan require 24 kHz audio.

    def embed(wav16: np.ndarray) -> np.ndarray:
        import torchaudio
        wav24 = torchaudio.functional.resample(
            torch.from_numpy(wav16).float().unsqueeze(0), SAMPLE_RATE, sr_model
        ).to(DEVICE)
        with torch.no_grad():
            emb = model(wavs=wav24)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = emb.squeeze(0)
        return emb.float().cpu().numpy().astype(np.float16)

    return embed, "muq"


def _load_moss_nano():
    """MOSS-Audio-Tokenizer-Nano: use encoder output before RVQ quantization."""
    from transformers import AutoModel
    name = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(DEVICE).eval()
    sr_model = int(getattr(model, "sampling_rate", 48_000))  # MOSS Nano uses 48 kHz stereo.
    n_channels = int(getattr(model.config, "number_channels", 2))

    def embed(wav16: np.ndarray) -> np.ndarray:
        import torchaudio
        wav48 = torchaudio.functional.resample(
            torch.from_numpy(wav16).float().unsqueeze(0), SAMPLE_RATE, sr_model
        )  # (1, samples)
        if n_channels > 1:
            wav48 = wav48.repeat(n_channels, 1)
        wav48 = wav48[:n_channels].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            enc_out = model.encode(wav48, return_dict=True)
        if enc_out.encoder_hidden_states is None:
            raise RuntimeError("MOSS encoder did not return encoder_hidden_states")
        emb = enc_out.encoder_hidden_states.squeeze(0).mean(-1)
        return emb.float().cpu().numpy().astype(np.float16)

    return embed, "moss_nano"


def _load_clap():
    from transformers import ClapModel, ClapProcessor
    name = "laion/larger_clap_music"
    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).half().to(DEVICE).eval()
    sr_model = 48_000

    def embed(wav16: np.ndarray) -> np.ndarray:
        import torchaudio
        wav48 = torchaudio.functional.resample(
            torch.from_numpy(wav16).unsqueeze(0), SAMPLE_RATE, sr_model
        ).squeeze(0).numpy()
        inputs = proc(audio=wav48, sampling_rate=sr_model,
                      return_tensors="pt", padding=True)
        inputs = {k: v.half().to(DEVICE) if v.dtype == torch.float32 else v.to(DEVICE)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model.get_audio_features(**inputs)
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
        return emb.squeeze(0).float().cpu().numpy().astype(np.float16)

    return embed, "clap"


def _load_encodec():
    from transformers import EncodecModel, AutoProcessor
    name = "facebook/encodec_24khz"
    proc = AutoProcessor.from_pretrained(name)
    model = EncodecModel.from_pretrained(name).half().to(DEVICE).eval()
    sr_model = 24_000

    def embed(wav16: np.ndarray) -> np.ndarray:
        import torchaudio
        wav24 = torchaudio.functional.resample(
            torch.from_numpy(wav16).unsqueeze(0), SAMPLE_RATE, sr_model
        ).numpy()
        inputs = proc(raw_audio=wav24, sampling_rate=sr_model, return_tensors="pt")
        inputs = {k: v.half().to(DEVICE) if v.dtype == torch.float32 else v.to(DEVICE)
                  for k, v in inputs.items()}
        with torch.no_grad():
            enc_out = model.encode(**inputs)
        # Use continuous encoder frame embeddings (before quantizer)
        emb = enc_out.audio_codes.float().squeeze().mean(-1)
        if emb.dim() == 0:
            emb = emb.unsqueeze(0)
        return emb.cpu().numpy().astype(np.float16)

    return embed, "encodec"


_LOADERS = {
    "mert": _load_mert,
    "muq": _load_muq,
    "moss_nano": _load_moss_nano,
    "clap": _load_clap,
    "encodec": _load_encodec,
}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="mert,muq,moss_nano,clap",
        help="Comma-separated list: mert,muq,moss_nano,clap,encodec",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    manifest = build_song_manifest()
    all_entries: list[dict] = []
    seen: set[str] = set()
    for split in ("training", "validation", "test", "test-ood"):
        for e in load_split(split, manifest, max_records=args.limit):
            if e["uuid"] not in seen:
                seen.add(e["uuid"])
                all_entries.append(e)

    print(f"Embedding {len(all_entries)} unique clips with models: {model_names}")

    for model_key in model_names:
        if model_key not in _LOADERS:
            print(f"  Unknown model '{model_key}', skipping.")
            continue
        print(f"\n── {model_key} ──")
        out_dir = EMBED_DIR / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            embed_fn, _ = _LOADERS[model_key]()
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] Could not load {model_key}: {exc}")
            continue

        for entry in tqdm(all_entries, desc=model_key, ncols=90):
            out_path = out_dir / f"{entry['uuid']}.npy"
            if out_path.exists():
                continue
            try:
                wav = load_audio_mono(entry["_audio_path"])
                emb = embed_fn(wav)
                np.save(out_path, emb)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERR] {entry['uuid']}: {exc}", flush=True)

        print(f"  {model_key} done → {out_dir}")


if __name__ == "__main__":
    main()
