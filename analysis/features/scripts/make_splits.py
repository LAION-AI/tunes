"""Materialize ``training``/``validation``/``test``/``test-ood`` from
``./data/songs.jsonl``.

Base split assignment is deterministic:

  * Rows with ``split_pool == "ood"``  →  ``test-ood``
  * Rows with ``is_held_out == True``  →  ``test``
  * Remaining rows                     →  ``training`` / ``validation``
    via a hash of ``(seed, song_id)`` so assignment is stable across runs.

Optional enhancement for OOD evaluation:
  * Promote a metadata-selected subset of human songs from ``test`` to
    ``test-ood`` (labelled by a ``ood_reason`` field), so OOD evaluation
    includes both classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SONGS_PATH = DATA_DIR / "songs.jsonl"

OUT_FILES = {
    "training": DATA_DIR / "training.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
    "test-ood": DATA_DIR / "test-ood.jsonl",
}


def iter_songs(path: Path = SONGS_PATH) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _bucket(seed: int, song_id: str) -> float:
    """Deterministic float in [0, 1) from (seed, song_id)."""
    h = hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_split(record: dict, *, seed: int, val_ratio: float) -> str:
    if record.get("split_pool") == "ood":
        return "test-ood"
    if record.get("is_held_out"):
        return "test"
    return "validation" if _bucket(seed, record["song_id"]) < val_ratio else "training"


def _safe_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def _safe_str(v: object) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _metadata_rarity_score(
    rec: dict, *, genre_freq: dict[str, int], lang_freq: dict[str, int], total: int
) -> float:
    """Higher means 'more OOD-like' within human metadata distribution."""
    meta = rec.get("meta", {})
    if not isinstance(meta, dict):
        return 0.0
    genres = _safe_list(meta.get("genres"))
    lang = _safe_str(meta.get("language"))

    score = 0.0
    # Rare language boosts score.
    if lang:
        p_lang = lang_freq.get(lang, 0) / max(total, 1)
        score += -math.log(max(p_lang, 1e-12))
    else:
        score += 3.0

    # Rare genres boost score; many genres slightly penalized.
    if genres:
        probs = [genre_freq.get(g, 0) / max(total, 1) for g in genres]
        rarity = sum(-math.log(max(p, 1e-12)) for p in probs) / len(probs)
        score += rarity - 0.1 * max(len(genres) - 3, 0)
    else:
        score += 2.0

    return score


def select_human_ood_from_metadata(
    rows: list[dict],
    *,
    base_splits: dict[str, str],
    seed: int,
    target_count: int,
) -> set[str]:
    """Pick human test rows with rare metadata and return song_ids."""
    if target_count <= 0:
        return set()

    human_train_val = [
        r
        for r in rows
        if r.get("source") == "human" and base_splits.get(r["song_id"]) in {"training", "validation"}
    ]
    human_test = [
        r
        for r in rows
        if r.get("source") == "human" and base_splits.get(r["song_id"]) == "test"
    ]
    if not human_test:
        return set()

    genre_freq: dict[str, int] = defaultdict(int)
    lang_freq: dict[str, int] = defaultdict(int)
    for rec in human_train_val:
        meta = rec.get("meta", {})
        if not isinstance(meta, dict):
            continue
        for g in _safe_list(meta.get("genres")):
            genre_freq[g] += 1
        lang = _safe_str(meta.get("language"))
        if lang:
            lang_freq[lang] += 1

    scored: list[tuple[float, float, str]] = []
    for rec in human_test:
        sid = rec["song_id"]
        score = _metadata_rarity_score(
            rec, genre_freq=genre_freq, lang_freq=lang_freq, total=max(len(human_train_val), 1)
        )
        # deterministic tie-breaker
        tie = _bucket(seed, sid)
        scored.append((score, tie, sid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = {sid for _, _, sid in scored[: min(target_count, len(scored))]}
    return selected


def materialize(
    rows: list[dict],
    *,
    seed: int,
    val_ratio: float,
    include_human_ood: bool,
    human_ood_count: int,
    out_files: dict[str, Path] = OUT_FILES,
) -> tuple[dict[str, dict[str, int]], int]:
    """Write split files; return per-split counts and human-ood promoted count."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_splits = {
        rec["song_id"]: assign_split(rec, seed=seed, val_ratio=val_ratio) for rec in rows
    }

    ai_ood_count = sum(
        1 for rec in rows if rec.get("split_pool") == "ood" and rec.get("source") != "human"
    )
    target_human_ood = human_ood_count if human_ood_count >= 0 else ai_ood_count
    human_ood_ids = (
        select_human_ood_from_metadata(
            rows, base_splits=base_splits, seed=seed, target_count=target_human_ood
        )
        if include_human_ood
        else set()
    )

    handles = {name: path.open("w", encoding="utf-8") for name, path in out_files.items()}
    counts: dict[str, dict[str, int]] = {n: defaultdict(int) for n in out_files}
    try:
        for rec in rows:
            split = base_splits[rec["song_id"]]
            out_rec = rec
            if rec["song_id"] in human_ood_ids and split == "test":
                split = "test-ood"
                out_rec = dict(rec)
                out_rec["ood_reason"] = "human_metadata_rarity"
            handles[split].write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            counts[split][rec["source"]] += 1
    finally:
        for h in handles.values():
            h.close()
    return {n: dict(d) for n, d in counts.items()}, len(human_ood_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation fraction of the in-distribution training pool.",
    )
    parser.add_argument(
        "--songs",
        type=Path,
        default=SONGS_PATH,
        help="Path to the unified songs.jsonl file.",
    )
    parser.add_argument(
        "--include-human-ood",
        action="store_true",
        help="Promote metadata-selected human test songs into test-ood.",
    )
    parser.add_argument(
        "--human-ood-count",
        type=int,
        default=-1,
        help=(
            "How many human songs to move from test -> test-ood. "
            "Default -1 matches AI OOD count."
        ),
    )
    args = parser.parse_args()

    if not args.songs.exists():
        raise FileNotFoundError(
            f"{args.songs} not found. Run scripts/build_data.py first."
        )

    rows = list(iter_songs(args.songs))
    counts, human_ood_promoted = materialize(
        rows,
        seed=args.seed,
        val_ratio=args.val,
        include_human_ood=args.include_human_ood,
        human_ood_count=args.human_ood_count,
    )

    sources_seen: set[str] = set()
    for c in counts.values():
        sources_seen.update(c)
    sources = sorted(sources_seen)

    print(f"\n=== Splits (seed={args.seed}, val={args.val}) ===")
    header = f"{'split':<14} " + " ".join(f"{s:>10}" for s in sources) + f" {'TOTAL':>10}"
    print(header)
    print("-" * len(header))
    for split, c in counts.items():
        total = sum(c.values())
        row = f"{split:<14} " + " ".join(f"{c.get(s, 0):>10}" for s in sources) + f" {total:>10}"
        print(row)

    if args.include_human_ood:
        print(
            f"\nHuman OOD promotion enabled: moved {human_ood_promoted} "
            "human songs from test -> test-ood"
        )

    print()
    for name, path in OUT_FILES.items():
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
