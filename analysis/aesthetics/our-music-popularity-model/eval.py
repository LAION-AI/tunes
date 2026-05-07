"""Run ``laion/music-popularity`` on a file, directory, txt list, or input.jsonl."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

from model_architecture import PopularityMLP

HERE = Path(__file__).resolve().parent
WHISPER_REPO = "laion/music-whisper"
POPULARITY_REPO = "laion/music-popularity"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4")
TARGET_SAMPLE_RATE = 16000
TARGET_SECONDS = 30
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_SECONDS


def is_audio_file(path: str | Path) -> bool:
    return str(path).lower().endswith(AUDIO_EXTENSIONS)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def safe_torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_models(device: torch.device):
    print("Loading Whisper encoder...")
    processor = WhisperProcessor.from_pretrained(WHISPER_REPO)
    whisper = WhisperModel.from_pretrained(WHISPER_REPO).encoder.to(device).eval()

    print("Loading popularity head...")
    model = PopularityMLP().to(device)
    head_path = hf_hub_download(repo_id=POPULARITY_REPO, filename="popularity_head.pt")
    state = safe_torch_load(head_path, device)
    if isinstance(state, dict) and "mlp_state_dict" in state:
        state = state["mlp_state_dict"]
    model.load_state_dict(state, strict=False)
    return processor, whisper, model.eval()


def load_audio_30s(audio_path: str | Path) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    audio = audio[:TARGET_SAMPLES]
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    return audio.astype(np.float32, copy=False)


@torch.no_grad()
def extract_embedding(audio_path: str | Path, processor, whisper, device: torch.device) -> torch.Tensor:
    audio = load_audio_30s(audio_path)
    inputs = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    outputs = whisper(inputs.input_features.to(device))
    last_hidden = outputs.last_hidden_state
    if last_hidden.shape[1:] != (1500, 768):
        raise ValueError(
            f"Unexpected Whisper hidden-state shape for {audio_path}: "
            f"{tuple(last_hidden.shape)}"
        )

    segments = last_hidden.view(1, 10, 150, 768)
    pooled = torch.cat(
        [
            segments.mean(2),
            segments.max(2).values,
            segments.min(2).values,
        ],
        dim=2,
    )
    return pooled.view(1, -1).float()


@torch.no_grad()
def predict_popularity(audio_path: str | Path, processor, whisper, model, device: torch.device) -> dict[str, float]:
    embedding = extract_embedding(audio_path, processor, whisper, device)
    pred_play, pred_upvote = model(embedding)
    log_play = float(pred_play.item())
    log_upvote = float(pred_upvote.item())
    return {
        "log1p_play_count": round(log_play, 4),
        "log1p_upvote_count": round(log_upvote, 4),
        "estimated_play_count": round(max(math.expm1(log_play), 0.0), 4),
        "estimated_upvote_count": round(max(math.expm1(log_upvote), 0.0), 4),
    }


def load_inputs(input_path: str | Path) -> list[dict[str, str]]:
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        rows = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            path = Path(str(row.get("path") or row.get("input_path") or ""))
            if not path.exists():
                continue
            rows.append({
                "song_id": str(row.get("song_id") or path.stem),
                "song_source": row.get("song_source"),
                "snippet_kind": row.get("snippet_kind"),
                "path": str(path),
            })
        return rows

    if input_path.is_file() and is_audio_file(input_path):
        return [{"song_id": input_path.stem, "path": str(input_path)}]

    if input_path.is_file():
        rows = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            path = Path(line.strip())
            if path.exists() and is_audio_file(path):
                rows.append({"song_id": path.stem, "path": str(path)})
        return rows

    if input_path.is_dir():
        return [
            {"song_id": Path(path).stem, "path": path}
            for path in glob.glob(str(input_path / "*"))
            if is_audio_file(path)
        ]

    raise ValueError(f"input_path {input_path} is not a file or directory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path",
        default=str(HERE / "input.jsonl"),
        help="Path to input.jsonl, txt file list, audio file, or audio directory.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=str(HERE / "output"),
        help="Output directory for result.json.",
    )
    parser.add_argument(
        "--use_cpu",
        type=str_to_bool,
        default=False,
        help="Force CPU mode even if a GPU is available.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if (torch.cuda.is_available() and not args.use_cpu) else torch.device("cpu")

    rows = load_inputs(args.input_path)
    print(f"input files: {len(rows)} from {args.input_path}")
    processor, whisper, model = load_models(device)

    result = {}
    metadata = {}
    for row in tqdm(rows):
        sid = str(row["song_id"])
        try:
            result[sid] = predict_popularity(row["path"], processor, whisper, model, device)
            metadata[sid] = {
                k: row.get(k)
                for k in ("song_source", "snippet_kind", "path")
                if row.get(k) is not None
            }
        except Exception as exc:
            print(f"failed {row['path']}: {exc}")

    (output_dir / "result.json").write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(result)} scores to {output_dir / 'result.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
