"""Build inputs for ``laion/music-aesthetics`` from the audiobox staging set.

This intentionally reuses ``analysis/aesthetics/audiobox/input.jsonl`` so all
audio-only aesthetics front-ends are evaluated on the exact same snippets.
Run ``analysis/aesthetics/audiobox/build_input.py`` first if the source file is
missing.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIOBOX_INPUT = HERE.parent / "audiobox" / "input.jsonl"
OUR_INPUT_JSONL = HERE / "input.jsonl"
OUR_INPUT_TXT = HERE / "input.txt"


def main() -> None:
    if not AUDIOBOX_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {AUDIOBOX_INPUT}. Run `uv run python aesthetics/audiobox/build_input.py` first."
        )

    rows = []
    for line in AUDIOBOX_INPUT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        path = Path(str(row.get("path") or ""))
        sid = str(row.get("song_id") or "").strip()
        if not sid or not path.exists():
            continue
        rows.append({
            "song_id": sid,
            "song_source": row.get("song_source"),
            "song_title": row.get("song_title"),
            "snippet_kind": row.get("snippet_kind"),
            "path": str(path),
        })

    OUR_INPUT_JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    OUR_INPUT_TXT.write_text(
        "".join(str(r["path"]) + "\n" for r in rows),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} rows to {OUR_INPUT_JSONL}")
    print(f"Wrote file list for eval.py to {OUR_INPUT_TXT}")


if __name__ == "__main__":
    main()
