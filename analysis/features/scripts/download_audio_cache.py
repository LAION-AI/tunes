"""Download every song referenced in ``./data/songs.jsonl`` to
``./data-cache/<source>/<uuid>.<ext>`` for reproducibility.

Why? Some upstream providers (Suno, Udio, ...) frequently rotate URLs
and remove songs. Persisting a local copy makes our experiments
replayable.

Layout::

    data-cache/
        human/
            <uuid>.m4a
            ...
        mureka/<uuid>.mp3
        sonauto/<uuid>.ogg
        suno/<uuid>.mp3
        udio/<uuid>.mp3
        lyria3/<uuid>.mp3
        riffusion/<uuid>.m4a
        silverknightai/<uuid>.mp3

A small ``manifest.jsonl`` is written alongside, recording which uuid
was downloaded from where, file size, and content hash. Re-running the
script is idempotent — already cached files are skipped (size match)
unless ``--force`` is given.

Run with::

    uv run python scripts/download_audio_cache.py
    uv run python scripts/download_audio_cache.py --workers 16 --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SONGS_PATH = DATA_DIR / "songs.jsonl"
CACHE_DIR = ROOT / "data-cache"
MANIFEST_PATH = CACHE_DIR / "manifest.jsonl"

DEFAULT_TIMEOUT = 60.0
ALLOWED_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".webm")


def iter_songs(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def guess_extension(url: str, fallback: str = ".mp3") -> str:
    path = urlparse(url).path
    suffix = Path(unquote(path)).suffix.lower()
    if suffix in ALLOWED_EXTS:
        return suffix
    return fallback


def download_one(
    client: httpx.Client,
    rec: dict,
    cache_dir: Path,
    force: bool,
) -> dict:
    source = rec["source"]
    uuid = rec["uuid"]
    url = rec["audio_url"]
    out_dir = cache_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = guess_extension(url)
    out_path = out_dir / f"{uuid}{ext}"

    result = {
        "song_id": rec["song_id"],
        "source": source,
        "uuid": uuid,
        "url": url,
        "path": str(out_path),
        "status": "skipped",
        "size": None,
        "sha256": None,
        "error": None,
    }

    if out_path.exists() and not force and out_path.stat().st_size > 0:
        result["status"] = "cached"
        result["size"] = out_path.stat().st_size
        return result

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    sha = hashlib.sha256()
    try:
        with client.stream("GET", url, timeout=DEFAULT_TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = 0
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    sha.update(chunk)
                    total += len(chunk)
        tmp_path.replace(out_path)
        result["status"] = "downloaded"
        result["size"] = total
        result["sha256"] = sha.hexdigest()
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=Path, default=SONGS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of songs to download (debug).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached file exists.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Limit to specific sources (e.g. --sources lyria3 riffusion).",
    )
    args = parser.parse_args()

    if not args.songs.exists():
        raise FileNotFoundError(
            f"{args.songs} not found. Run scripts/build_data.py first."
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    songs = list(iter_songs(args.songs))
    if args.sources:
        sel = set(args.sources)
        songs = [r for r in songs if r["source"] in sel]
    if args.limit:
        songs = songs[: args.limit]

    print(f"Downloading {len(songs)} songs into {args.cache_dir} with {args.workers} workers (force={args.force})")

    by_status = {"downloaded": 0, "cached": 0, "error": 0}
    by_source: dict[str, dict[str, int]] = {}
    errors: list[dict] = []
    total_bytes = 0
    started = time.time()

    headers = {"User-Agent": "songrating-experiments/0.1 (+local cache)"}
    manifest_path = args.cache_dir / "manifest.jsonl"
    manifest_fh = manifest_path.open("w", encoding="utf-8")
    try:
        with httpx.Client(headers=headers, http2=False, follow_redirects=True) as client:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(download_one, client, rec, args.cache_dir, args.force)
                    for rec in songs
                ]
                for fut in tqdm(as_completed(futures), total=len(futures), unit="song"):
                    res = fut.result()
                    manifest_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    by_status[res["status"]] = by_status.get(res["status"], 0) + 1
                    bucket = by_source.setdefault(
                        res["source"], {"downloaded": 0, "cached": 0, "error": 0}
                    )
                    bucket[res["status"]] = bucket.get(res["status"], 0) + 1
                    if res["size"]:
                        total_bytes += res["size"]
                    if res["status"] == "error":
                        errors.append(res)
    finally:
        manifest_fh.close()

    elapsed = time.time() - started
    print("\n=== Summary ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Total bytes: {total_bytes / 1e9:.2f} GB")
    print(f"Status totals: {by_status}")
    print()
    header = f"{'source':<18} {'downloaded':>12} {'cached':>10} {'error':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    for source, c in sorted(by_source.items()):
        total = c["downloaded"] + c["cached"] + c["error"]
        print(f"{source:<18} {c['downloaded']:>12} {c['cached']:>10} "
              f"{c['error']:>8} {total:>8}")

    manifest_path = args.cache_dir / "manifest.jsonl"
    print(f"\nManifest written to {manifest_path}")
    if errors:
        sample = errors[:5]
        print(f"\nFirst {len(sample)} errors (of {len(errors)}):")
        for e in sample:
            print(f"  [{e['source']}] {e['uuid']}: {e['error']}")


if __name__ == "__main__":
    main()
