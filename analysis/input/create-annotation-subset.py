"""
Creates annotation-subset-without-v1-or-v2 by copying annotation-subset
and removing all songs that appear in the v2 annotations export.
(v1 is a subset of v2, so filtering v2 covers both.)
"""

import json
from pathlib import Path

BASE = Path(__file__).parent

SRC_DIR = BASE / "annotation-subset"
DST_DIR = BASE / "annotation-subset-without-v1-or-v2"
V2_ANNOTATIONS = BASE / "v2" / "annotations_export_2026-04-20.json"

ID_FIELD = {
    "human_songs.jsonl": "uuid",
    "suno_subset.jsonl": "id",
    "udio_subset.jsonl": "id",
    "sonauto_subset.jsonl": "id",
    "mureka_subset.jsonl": "uuid",
}

SOURCE_MAP = {
    "human_songs.jsonl": "human",
    "suno_subset.jsonl": "suno",
    "udio_subset.jsonl": "udio",
    "sonauto_subset.jsonl": "sonauto",
    "mureka_subset.jsonl": "mureka",
}

with open(V2_ANNOTATIONS) as f:
    v2_annotations = json.load(f)

v2_ids = {}
for annotation in v2_annotations:
    source = annotation["song_source"]
    song_uuid = annotation["song_id"].split("_", 1)[1]
    v2_ids.setdefault(source, set()).add(song_uuid)

DST_DIR.mkdir(exist_ok=True)

for filename, id_field in ID_FIELD.items():
    src_file = SRC_DIR / filename
    dst_file = DST_DIR / filename
    source = SOURCE_MAP[filename]
    excluded_ids = v2_ids.get(source, set())
    kept = 0
    removed = 0
    with open(src_file) as f_in, open(dst_file, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            song = json.loads(line)
            if song[id_field] in excluded_ids:
                removed += 1
            else:
                f_out.write(line + "\n")
                kept += 1
    print(f"{filename}: kept {kept}, removed {removed}")
