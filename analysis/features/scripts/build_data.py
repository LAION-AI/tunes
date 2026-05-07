"""Combine all per-provider dataset files into a single ``./data/songs.jsonl``.

Each row of ``songs.jsonl`` has a normalized envelope so that downstream
loaders / training code do not need provider-specific glue:

    {
        "uuid": "...",                 # provider-native uuid
        "song_id": "<source>_<uuid>",  # canonical id (matches subtract-songs.json)
        "source": "human" | "mureka" | "sonauto" | "suno" | "udio" |
                   "lyria3" | "riffusion" | "silverknightai",
        "label": "human" | "ai",
        "split_pool": "id" | "ood",    # pool a row belongs to
        "is_held_out": bool,            # listed in subtract-songs.json
        "audio_url": "...",             # normalized download url
        "title": "...",
        "model": "...",
        "duration_ms": int | null,
        "meta": { ...original record fields... }
    }

The unified file is the single source of truth. The actual
training/validation/test/test-ood splits are produced on the fly from a
fixed seed by ``scripts/make_splits.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
DATA_DIR = ROOT / "data"
SUBTRACT_PATH = ROOT / "substract-songs.json"

OUT_PATH = DATA_DIR / "songs.jsonl"

ID_SOURCES = ("human", "mureka", "sonauto", "suno", "udio")
OOD_SOURCES = ("lyria3", "riffusion", "silverknightai")
HUMAN_SOURCES = {"human"}

SOURCE_FILES = {
    "human": "human_songs.jsonl",
    "mureka": "mureka_subset.jsonl",
    "sonauto": "sonauto_subset.jsonl",
    "suno": "suno_subset.jsonl",
    "udio": "udio_subset.jsonl",
    "lyria3": "lyria_subset.jsonl",
    "riffusion": "riffusion_subset.jsonl",
    "silverknightai": "silverknightai_subset.jsonl",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_record(source: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Project a provider-native row into the unified schema."""
    if source in ("human", "mureka"):
        uuid = raw["uuid"]
        title = raw.get("title")
        audio = raw.get("audio_url")
        model = raw.get("model")
        duration = raw.get("duration in milliseconds")
    elif source == "sonauto":
        uuid = raw["id"]
        title = raw.get("title")
        audio = raw.get("song_path")
        model = "sonauto"
        duration = None
    elif source == "suno":
        uuid = raw["id"]
        title = raw.get("title")
        audio = raw.get("audio_url")
        model = raw.get("model_name") or raw.get("major_model_version")
        duration = None
    elif source == "udio":
        uuid = raw["id"]
        title = raw.get("title")
        audio = raw.get("song_path")
        model = "udio"
        dur_s = raw.get("duration")
        duration = int(dur_s * 1000) if isinstance(dur_s, (int, float)) else None
    elif source in OOD_SOURCES:
        uuid = raw["uuid"]
        title = raw.get("title")
        audio = raw.get("audio_url")
        model = raw.get("model")
        duration = raw.get("duration in milliseconds")
    else:
        raise ValueError(f"Unknown source: {source}")

    return {
        "uuid": uuid,
        "song_id": f"{source}_{uuid}",
        "source": source,
        "label": "human" if source in HUMAN_SOURCES else "ai",
        "split_pool": "ood" if source in OOD_SOURCES else "id",
        "audio_url": audio,
        "title": title,
        "model": model,
        "duration_ms": duration,
        "meta": raw,
    }


def load_holdout_ids() -> set[str]:
    if not SUBTRACT_PATH.exists():
        return set()
    with SUBTRACT_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["song_id"] for item in items if item.get("song_id")}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    holdout = load_holdout_ids()
    print(f"Loaded {len(holdout)} held-out song_ids from subtract-songs.json")

    counts: dict[str, dict[str, int]] = {}
    written = 0
    skipped_no_audio = 0
    holdout_matched: set[str] = set()

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for source, fname in SOURCE_FILES.items():
            path = DATASETS_DIR / fname
            if not path.exists():
                print(f"  WARNING: missing {path}, skipping {source}")
                counts[source] = {"available": 0, "kept": 0, "held_out": 0}
                continue
            available = kept = held = 0
            for raw in iter_jsonl(path):
                available += 1
                rec = normalize_record(source, raw)
                rec["is_held_out"] = rec["song_id"] in holdout
                if not rec["audio_url"]:
                    skipped_no_audio += 1
                    continue
                if rec["is_held_out"]:
                    held += 1
                    holdout_matched.add(rec["song_id"])
                kept += 1
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            counts[source] = {"available": available, "kept": kept, "held_out": held}

    print("\n=== Summary ===")
    print(f"{'source':<18} {'available':>10} {'kept':>8} {'held_out':>10}")
    for source, c in counts.items():
        print(
            f"{source:<18} {c['available']:>10} {c['kept']:>8} {c['held_out']:>10}"
        )
    print(f"{'TOTAL':<18} "
          f"{sum(c['available'] for c in counts.values()):>10} "
          f"{sum(c['kept'] for c in counts.values()):>8} "
          f"{sum(c['held_out'] for c in counts.values()):>10}")
    print(f"\nRows skipped due to missing audio_url: {skipped_no_audio}")
    print(
        f"subtract-songs.json entries matched: "
        f"{len(holdout_matched)} / {len(holdout)}"
    )
    if missing := holdout - holdout_matched:
        print(f"  WARNING: {len(missing)} held-out ids not present in datasets")
    print(f"\nWrote unified file: {OUT_PATH.relative_to(ROOT)} ({written} rows)")


if __name__ == "__main__":
    main()
