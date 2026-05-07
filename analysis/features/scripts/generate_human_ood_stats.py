"""Generate HUMAN-OOD-STATS.md from current split files.

The report focuses on human songs and compares:
  - human rows in test-ood (with ood_reason)
  - human rows in training/validation/test (non-ood reference)

It highlights metadata values that appear only in human OOD, including
languages and genres.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "HUMAN-OOD-STATS.md"

SPLIT_FILES = {
    "training": DATA_DIR / "training.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
    "test-ood": DATA_DIR / "test-ood.jsonl",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def safe_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def top_lines(counter: Counter[str], n: int = 20) -> list[str]:
    if not counter:
        return ["- none"]
    lines: list[str] = []
    for key, cnt in counter.most_common(n):
        lines.append(f"- `{key}`: {cnt}")
    return lines


def only_in_a(a: Counter[str], b: Counter[str]) -> Counter[str]:
    return Counter({k: v for k, v in a.items() if k not in b})


def main() -> None:
    missing = [p for p in SPLIT_FILES.values() if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing split files: {missing_list}. Run scripts/make_splits.py first."
        )

    human_ood: list[dict] = []
    human_non_ood: list[dict] = []

    for split, path in SPLIT_FILES.items():
        for row in iter_jsonl(path):
            if row.get("source") != "human":
                continue
            if split == "test-ood":
                human_ood.append(row)
            else:
                human_non_ood.append(row)

    ood_lang = Counter()
    non_ood_lang = Counter()
    ood_genres = Counter()
    non_ood_genres = Counter()
    ood_artists = Counter()
    non_ood_artists = Counter()
    ood_albums = Counter()
    non_ood_albums = Counter()
    ood_reasons = Counter()

    def consume(rows: list[dict], lang: Counter, genres: Counter, artists: Counter, albums: Counter) -> None:
        for rec in rows:
            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                continue
            lg = safe_str(meta.get("language"))
            if lg:
                lang[lg] += 1
            for g in safe_list(meta.get("genres")):
                genres[g] += 1
            artist = safe_str(meta.get("primary_artist"))
            if artist:
                artists[artist] += 1
            album = safe_str(meta.get("album_name"))
            if album:
                albums[album] += 1

    consume(human_ood, ood_lang, ood_genres, ood_artists, ood_albums)
    consume(human_non_ood, non_ood_lang, non_ood_genres, non_ood_artists, non_ood_albums)

    for rec in human_ood:
        reason = safe_str(rec.get("ood_reason")) or "unspecified"
        ood_reasons[reason] += 1

    unique_ood_lang = only_in_a(ood_lang, non_ood_lang)
    unique_ood_genres = only_in_a(ood_genres, non_ood_genres)
    unique_ood_artists = only_in_a(ood_artists, non_ood_artists)
    unique_ood_albums = only_in_a(ood_albums, non_ood_albums)

    lines: list[str] = []
    lines.append("# HUMAN OOD Stats")
    lines.append("")
    lines.append("Generated from current `data/*.jsonl` split files.")
    lines.append("")
    lines.append("## Population")
    lines.append("")
    lines.append(f"- Human rows in `test-ood`: **{len(human_ood)}**")
    lines.append(f"- Human rows in non-OOD (`training` + `validation` + `test`): **{len(human_non_ood)}**")
    lines.append("")
    lines.append("## OOD Reason Breakdown")
    lines.append("")
    lines.extend(top_lines(ood_reasons, n=20))
    lines.append("")
    lines.append("## Languages In Human OOD")
    lines.append("")
    lines.append("### Top languages in human OOD")
    lines.append("")
    lines.extend(top_lines(ood_lang, n=20))
    lines.append("")
    lines.append("### Languages only in human OOD (not in non-OOD human)")
    lines.append("")
    lines.extend(top_lines(unique_ood_lang, n=50))
    lines.append("")
    lines.append("## Genres In Human OOD")
    lines.append("")
    lines.append("### Top genres in human OOD")
    lines.append("")
    lines.extend(top_lines(ood_genres, n=50))
    lines.append("")
    lines.append("### Genres only in human OOD (not in non-OOD human)")
    lines.append("")
    lines.extend(top_lines(unique_ood_genres, n=100))
    lines.append("")
    lines.append("## Additional Metadata Only In Human OOD")
    lines.append("")
    lines.append("### Artists only in human OOD")
    lines.append("")
    lines.extend(top_lines(unique_ood_artists, n=50))
    lines.append("")
    lines.append("### Albums only in human OOD")
    lines.append("")
    lines.extend(top_lines(unique_ood_albums, n=50))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report compares metadata values by exact string match.")
    lines.append("- If you rerun `scripts/make_splits.py`, regenerate this file.")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(
        "Summary: "
        f"human_ood={len(human_ood)}, "
        f"unique_ood_languages={len(unique_ood_lang)}, "
        f"unique_ood_genres={len(unique_ood_genres)}"
    )


if __name__ == "__main__":
    main()
