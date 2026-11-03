"""
A5 — Do the *released* embeddings carry usable signal?

Addresses reviewer N3D6-W3: "The released features, such as the 768-dimensional
embedding and the embeddings extracted from the ASR model, are not verified.
How much can we trust these features if no experiment validates them?"

We answer with linear probes on the embeddings exactly as a downstream user
receives them, pulled directly from the public release
(huggingface.co/datasets/laion/laion-tunes): the 768-d Music-Whisper audio
fingerprints in `whisper_embeddings/*.npz` joined by `filename` to the
per-track metadata in `public/*.tar.parquet`.

A *linear* probe is the right instrument: it cannot manufacture structure that
the representation does not already expose linearly, so beating baselines means
the shipped vectors are directly useful, not merely that a big head could be
trained on them.

Probe targets are chosen to be independent of the embedding itself:
  P1  Vocal vs instrumental (`has_lyrics`) -- derived from Parakeet ASR, a
      different model than the embedding.
  P2  Source platform (Suno / Mureka / Udio) -- crawl provenance, not audio.
  P3  Primary genre tag -- user-supplied text on the platform.
  P4  Source platform from the released Parakeet transcription embeddings.
  P5  Primary genre tag from the released transcription embeddings.
      (P4/P5 are populated only where vocals were detected, so n is smaller.)

Deliberately NOT probed: the released `score_*` aesthetics fields, because the
aesthetics head is itself trained on these very embeddings, so recovering them
would be circular.

Controls: majority-class baseline and a shuffled-label run (the latter must
collapse to baseline, confirming the pipeline has no leakage).

Outputs
  output/feature_utility_probe.csv
  output/feature_utility_probe_table.tex

Run:  .venv/bin/python feature_utility_probe.py
      .venv/bin/python feature_utility_probe.py --table-only   (rebuild the
      LaTeX table from output/feature_utility_probe.csv without re-downloading)
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
CACHE = BASE / ".cache_a5"
REPO = "https://huggingface.co/datasets/laion/laion-tunes/resolve/main/"
SEED = 42

# Shards to pull. Kept modest: enough tracks for a stable linear probe while
# staying a light download against the public release.
SHARDS = {
    "udio":   {"npz": ["udio_000000", "udio_000001"], "parquet": [f"udio_{i:06d}" for i in range(12)]},
    "mureka": {"npz": ["mureka_000000"],              "parquet": [f"mureka_{i:06d}" for i in range(3)]},
    "suno":   {"npz": ["suno_000029", "suno_000049"], "parquet": ["suno_000029", "suno_000049"]},
}


def fetch(path: str) -> bytes:
    """Download `path` from the public release, caching locally."""
    dest = CACHE / path.replace("/", "_")
    if dest.exists():
        return dest.read_bytes()
    r = requests.get(REPO + path, timeout=600)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return r.content


def primary_genre(genre_tags, tags_text) -> str | None:
    """First parsed genre tag, else None."""
    if genre_tags is not None:
        if isinstance(genre_tags, np.ndarray):
            genre_tags = genre_tags.tolist()
        if isinstance(genre_tags, list) and genre_tags:
            g = str(genre_tags[0]).strip().lower()
            if g:
                return g
    return None


def load_platform(platform: str) -> pd.DataFrame:
    """Join released embeddings to released metadata for one platform."""
    emb: dict[str, np.ndarray] = {}
    for shard in SHARDS[platform]["npz"]:
        z = np.load(io.BytesIO(fetch(f"whisper_embeddings/{shard}.npz")), allow_pickle=True)
        for fn, v in zip(z["filenames"], z["embeddings"]):
            emb[str(fn)] = v

    rows = []
    for shard in SHARDS[platform]["parquet"]:
        try:
            raw = fetch(f"public/{shard}.tar.parquet")
        except requests.HTTPError:
            continue
        t = pq.read_table(io.BytesIO(raw))
        cols = set(t.schema.names)
        for rec in t.to_pylist():
            fn = str(rec["filename"])
            if fn not in emb:
                continue
            # Released ASR/transcription-side vector. Reviewer N3D6 names this
            # separately from the audio embedding, so it gets its own probes.
            tr = rec.get("lyric_embedding") if "lyric_embedding" in cols else None
            if tr is not None and len(tr) > 0:
                a = np.asarray(tr, dtype=np.float64)
                tr = a if np.abs(a).sum() > 0 else None   # unpopulated for instrumentals
            else:
                tr = None
            hl = rec.get("has_lyrics") if "has_lyrics" in cols else None
            # Coerce to a plain Python bool/None: parquet values can arrive
            # Arrow-backed, which sklearn cannot index with integer arrays.
            hl = None if hl is None else bool(hl)
            rows.append({
                "filename": fn,
                "platform": str(platform),
                "embedding": emb[fn],
                "transcription_embedding": tr,
                "has_lyrics": hl,
                "genre": primary_genre(rec.get("genre_tags"), rec.get("tags_text")),
            })
    df = pd.DataFrame(rows)
    df["has_lyrics"] = df["has_lyrics"].astype("object")
    print(f"  {platform:7s} embeddings={len(emb):5d}  joined tracks={len(df):5d}")
    return df


def probe(X: np.ndarray, y, name: str, records: list, balance: bool = False) -> None:
    """Linear probe with majority + shuffled-label controls.

    `y` is coerced to a plain object ndarray: metadata read from parquet can be
    Arrow-backed, which sklearn cannot index with integer arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(pd.Series(y).astype(str).to_numpy(), dtype=object)

    if balance:
        # Cap every class at the smallest class so accuracy is interpretable
        # against a 1/k baseline rather than dominated by one platform.
        rng = np.random.default_rng(SEED)
        counts = Counter(y)
        cap = min(counts.values())
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y == c))[:cap] for c in sorted(counts)
        ])
        keep.sort()
        X, y = X[keep], y[keep]
        print(f"  (balanced to {cap} per class -> n={len(y)})")

    classes = Counter(y)
    if len(classes) < 2 or min(classes.values()) < 10:
        print(f"  {name:34s} SKIPPED (insufficient class support: {dict(classes)})")
        return
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    # Majority-class baseline on the same test split. Its macro F1 matters as
    # much as its accuracy: on a skewed target (e.g. 89% vocal) accuracy alone
    # makes a trivial predictor look strong.
    maj = Counter(ytr).most_common(1)[0][0]
    maj_pred = np.full_like(yte, maj, dtype=object)
    base_acc = accuracy_score(yte, maj_pred)
    base_f1 = f1_score(yte, maj_pred, average="macro")

    # Shuffled-label control: must collapse to ~baseline if there is no leakage.
    rng = np.random.default_rng(SEED)
    ysh = rng.permutation(ytr)
    clf_sh = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr_s, ysh)
    sh_acc = accuracy_score(yte, clf_sh.predict(Xte_s))

    lift = acc - base_acc
    print(f"  {name:34s} acc={acc:.3f} (maj {base_acc:.3f})  macroF1={f1:.3f} "
          f"(maj {base_f1:.3f})  shuffled={sh_acc:.3f}  (n={len(y)}, {len(classes)} cls)")
    records.append({
        "probe": name, "n": int(len(y)), "n_classes": len(classes),
        "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
        "majority_baseline": round(float(base_acc), 4),
        "majority_macro_f1": round(float(base_f1), 4),
        "shuffled_control": round(float(sh_acc), 4),
        "lift_over_majority": round(float(lift), 4),
        "macro_f1_lift": round(float(f1 - base_f1), 4),
    })


GROUPS = (
    ("768-d Music-Whisper audio fingerprint", ("P1", "P2", "P3")),
    ("Parakeet transcription embedding", ("P4", "P5")),
)


def write_table(df_out: pd.DataFrame) -> None:
    """LaTeX table, grouped by embedding family (audio fingerprint, then transcription)."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Accuracy}} & \multicolumn{2}{c}{\textbf{Macro $F_1$}} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\textbf{Linear probe} & \textbf{$n$} & \textbf{Probe} & \textbf{Majority}"
        r" & \textbf{Probe} & \textbf{Majority} & \textbf{Shuf.} \\",
    ]
    for title, ids in GROUPS:
        sub = df_out[df_out.probe.str[:2].isin(ids)].sort_values("probe")
        if sub.empty:
            continue
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for _, r in sub.iterrows():
            lines.append(f"{r.probe} & {int(r.n)} & \\textbf{{{r.accuracy:.3f}}} & {r.majority_baseline:.3f} & "
                         f"\\textbf{{{r.macro_f1:.3f}}} & {r.majority_macro_f1:.3f} & {r.shuffled_control:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "feature_utility_probe_table.tex").write_text("\n".join(lines) + "\n")
    print("Wrote output/feature_utility_probe_table.tex")


def main() -> None:
    print("Loading released embeddings + metadata from the public HF release...")
    parts = [load_platform(p) for p in SHARDS]
    df = pd.concat(parts, ignore_index=True)
    X = np.vstack(df.embedding.values)
    print(f"\nJoined {len(df)} tracks; embedding matrix {X.shape}")
    print(f"L2 norms: mean={np.linalg.norm(X, axis=1).mean():.3f} "
          f"(released fingerprints are L2-normalized)")

    records: list[dict] = []

    print("\n" + "=" * 78)
    print("Linear probes on the released 768-d Music-Whisper audio fingerprint")
    print("=" * 78)

    # P1 vocal vs instrumental
    m = df.has_lyrics.notna()
    if m.sum() > 50:
        labels = ["vocal" if bool(v) else "instrumental" for v in df.loc[m, "has_lyrics"]]
        probe(X[m.values], labels, "P1 vocal vs instrumental", records)
    else:
        print("  P1 skipped: has_lyrics unavailable")

    # P2 platform (class-balanced: shard coverage is uneven across platforms)
    probe(X, df.platform.values, "P2 source platform (3-class)", records, balance=True)

    # P4/P5: the released ASR/transcription embeddings, which reviewer N3D6 names
    # separately. Populated only for tracks with detected vocals, so n is smaller
    # and the subset is platform-skewed; both are reported.
    tm = df.transcription_embedding.notna()
    if tm.sum() > 100:
        Xt = np.vstack(df.loc[tm, "transcription_embedding"].values)
        print(f"\n  -- transcription/ASR embeddings: {int(tm.sum())} populated tracks, "
              f"{dict(Counter(df.loc[tm, 'platform']))} --")
        probe(Xt, df.loc[tm, "platform"].values,
              "P4 platform from transcription emb.", records, balance=True)
        gt = tm & df.genre.notna()
        if gt.sum() > 100:
            cnt = Counter(df.loc[gt, "genre"])
            keep = {k for k, v in cnt.items() if v >= 40}
            gt2 = gt & df.genre.isin(keep)
            if gt2.sum() > 100 and len(keep) >= 2:
                Xg = np.vstack(df.loc[gt2, "transcription_embedding"].values)
                probe(Xg, df.loc[gt2, "genre"].values,
                      f"P5 genre from transcription emb. ({len(keep)}-class)", records)
    else:
        print("\n  transcription-embedding probes skipped: too few populated rows")

    # P3 primary genre, restricted to reasonably supported classes
    g = df.genre.notna()
    if g.sum() > 100:
        counts = Counter(df.loc[g, "genre"])
        keep = {k for k, v in counts.items() if v >= 40}
        gm = g & df.genre.isin(keep)
        if gm.sum() > 100 and len(keep) >= 2:
            print(f"  (genre probe restricted to {len(keep)} tags with n>=40: "
                  f"{sorted(keep)})")
            probe(X[gm.values], df.loc[gm, "genre"].values,
                  f"P3 primary genre ({len(keep)}-class)", records)
        else:
            print("  P3 skipped: too few well-supported genre classes")
    else:
        print("  P3 skipped: genre tags unavailable")

    df_out = pd.DataFrame(records)
    OUTPUT.mkdir(exist_ok=True)
    df_out.to_csv(OUTPUT / "feature_utility_probe.csv", index=False)
    print(f"\nWrote output/feature_utility_probe.csv")

    write_table(df_out)


if __name__ == "__main__":
    import sys
    if "--table-only" in sys.argv:
        write_table(pd.read_csv(OUTPUT / "feature_utility_probe.csv"))
    else:
        main()
