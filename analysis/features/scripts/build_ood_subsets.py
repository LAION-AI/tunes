"""Build OOD subsets for Lyria3, Riffusion, and SilverknightAI.

Outputs (in ``./datasets``):
    - lyria_subset.jsonl          (all available Lyria3 songs)
    - silverknightai_subset.jsonl (all available AICover songs)
    - riffusion_subset.jsonl      (sampled from sleeping-ai/Rdiffusion-audio)

Important: ``riffusion_subset.jsonl`` is built only from the audio repo
(``sleeping-ai/Rdiffusion-audio``) to avoid uncertain joins with external
metadata files.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid as uuid_lib
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RIFFUSION_INDEX_PATH = ROOT / "datasets" / ".riffusion_audio_index.json"

LYRIA_REPO = "sleeping-ai/Lyria3"
SILVERKNIGHT_REPO = "sleeping-ai/AICover"
RIFFUSION_AUDIO_REPO = "sleeping-ai/Rdiffusion-audio"

HF_API = "https://huggingface.co/api/datasets/{repo}/tree/main"
HF_API_PATH = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

SEED = 42
DEFAULT_RIFFUSION_COUNT = 51
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def list_hf_audio_files(repo: str) -> list[dict[str, Any]]:
    """Return audio file entries from a HuggingFace dataset's main tree."""
    url = HF_API.format(repo=repo)
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    entries = resp.json()
    return [
        e
        for e in entries
        if e.get("type") == "file"
        and e.get("path", "").lower().endswith(AUDIO_EXTS)
    ]


def list_hf_dir(repo: str, path: str) -> list[dict[str, Any]]:
    url = HF_API_PATH.format(repo=repo, path=path)
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def build_riffusion_audio_index(force: bool = False) -> dict[str, str]:
    """Map riffusion uuid -> repo path under sleeping-ai/Rdiffusion-audio."""
    if RIFFUSION_INDEX_PATH.exists() and not force:
        with RIFFUSION_INDEX_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Indexing audio files in {RIFFUSION_AUDIO_REPO} (cached afterwards)...")
    top = httpx.get(
        HF_API.format(repo=RIFFUSION_AUDIO_REPO), timeout=60.0, follow_redirects=True
    )
    top.raise_for_status()
    dirs = sorted(e["path"] for e in top.json() if e.get("type") == "directory")
    print(f"  found {len(dirs)} subdirectories")

    index: dict[str, str] = {}
    for d in dirs:
        files = list_hf_dir(RIFFUSION_AUDIO_REPO, d)
        added = 0
        for entry in files:
            if entry.get("type") != "file":
                continue
            p = entry.get("path", "")
            if not p.lower().endswith(AUDIO_EXTS):
                continue
            uid = Path(p).stem
            index[uid] = p
            added += 1
        print(f"  {d}: {added} audio files (running total: {len(index)})")

    RIFFUSION_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RIFFUSION_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"  cached index -> {RIFFUSION_INDEX_PATH.relative_to(ROOT)}")
    return index


def stable_uuid(namespace: str, name: str) -> str:
    """Deterministic UUID derived from namespace + name."""
    return str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, f"{namespace}/{name}"))


def lyria_records(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for entry in files:
        path = entry["path"]
        title = Path(path).stem
        records.append(
            {
                "uuid": stable_uuid(LYRIA_REPO, path),
                "title": title,
                "model": "Lyria3",
                "provider": "lyria3",
                "duration in milliseconds": None,
                "audio_url": HF_RESOLVE.format(repo=LYRIA_REPO, path=quote(path)),
                "size_bytes": entry.get("size"),
                "source_repo": LYRIA_REPO,
                "source_path": path,
            }
        )
    return records


def silverknight_records(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for entry in files:
        path = entry["path"]
        title = Path(path).stem
        records.append(
            {
                "uuid": stable_uuid(SILVERKNIGHT_REPO, path),
                "title": title,
                "model": "SilverknightAI",
                "provider": "silverknightai",
                "duration in milliseconds": None,
                "audio_url": HF_RESOLVE.format(repo=SILVERKNIGHT_REPO, path=quote(path)),
                "size_bytes": entry.get("size"),
                "source_repo": SILVERKNIGHT_REPO,
                "source_path": path,
            }
        )
    return records


def riffusion_records_from_audio_repo(
    index: dict[str, str], *, seed: int, limit: int | None
) -> list[dict[str, Any]]:
    items = sorted(index.items())
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:limit]

    records: list[dict[str, Any]] = []
    for uid, path in items:
        records.append(
            {
                "uuid": uid,
                "title": Path(path).stem,
                "model": "riffusion",
                "provider": "riffusion",
                "duration in milliseconds": None,
                "audio_url": HF_RESOLVE.format(repo=RIFFUSION_AUDIO_REPO, path=quote(path)),
                "source_repo": RIFFUSION_AUDIO_REPO,
                "source_path": path,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--riffusion-limit",
        type=int,
        default=DEFAULT_RIFFUSION_COUNT,
        help=(
            "How many Rdiffusion-audio songs to include. "
            "Default 51 picks a random subset for test-ood."
        ),
    )
    parser.add_argument(
        "--rebuild-riffusion-index",
        action="store_true",
        help="Force re-indexing of sleeping-ai/Rdiffusion-audio.",
    )
    args = parser.parse_args()

    print("Listing Lyria3 audio files on HuggingFace...")
    lyria_files = sorted(list_hf_audio_files(LYRIA_REPO), key=lambda e: e["path"])
    print(f"  found {len(lyria_files)} audio files")

    print("Listing SilverknightAI (AICover) audio files on HuggingFace...")
    silver_files = sorted(list_hf_audio_files(SILVERKNIGHT_REPO), key=lambda e: e["path"])
    print(f"  found {len(silver_files)} audio files")

    riffusion_index = build_riffusion_audio_index(force=args.rebuild_riffusion_index)
    riff_limit = args.riffusion_limit if args.riffusion_limit and args.riffusion_limit > 0 else None

    lyria = lyria_records(lyria_files)  # all Lyria3 songs
    silver = silverknight_records(silver_files)  # all AICover songs
    riffusion = riffusion_records_from_audio_repo(
        riffusion_index, seed=args.seed, limit=riff_limit
    )

    out_lyria = DATASETS_DIR / "lyria_subset.jsonl"
    out_silver = DATASETS_DIR / "silverknightai_subset.jsonl"
    out_riffusion = DATASETS_DIR / "riffusion_subset.jsonl"

    write_jsonl(out_lyria, lyria)
    write_jsonl(out_silver, silver)
    write_jsonl(out_riffusion, riffusion)

    print("\n=== Summary ===")
    print(f"  Lyria3 available:              {len(lyria_files):>6} -> {len(lyria):>6} written")
    print(f"  SilverknightAI available:      {len(silver_files):>6} -> {len(silver):>6} written")
    print(f"  Rdiffusion-audio indexed:      {len(riffusion_index):>6}")
    print(f"  Rdiffusion-audio selected:     {len(riffusion):>6}")
    if riff_limit is not None:
        print(f"  Riffusion limit used:          {riff_limit:>6}")
    else:
        print("  Riffusion limit used:             ALL")

    print(f"\nWrote: {out_lyria.relative_to(ROOT)}")
    print(f"Wrote: {out_silver.relative_to(ROOT)}")
    print(f"Wrote: {out_riffusion.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
