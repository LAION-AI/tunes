"""
A6 — Integrity of the human (Apple Music preview) control set.

Addresses reviewer Uhv7-Q3: "how do you rule out that some [controls] contain
AI-generated elements (which would contaminate the provenance labels
underpinning the detection and halo analyses)?"

Approach (metadata only, no audio needed):
  1. Release-date bound. Every control carries the album release date from its
     Genius metadata. A control released before 2023-01-01 predates public
     availability of every full-song generator in this study (Suno, Udio,
     Mureka, Sonauto), so it cannot be an output of them. Controls released in
     2023 or later, or without a date, are the only candidates for
     contamination; their count is an upper bound.
  2. Sensitivity. Re-compute the control-dependent headline quantities on the
     pre-2023 controls only: false-alarm rate on human songs, pooled d' (AI hit
     rate unchanged), balanced accuracy, and the perceived-label rating gap
     within actually-human songs (halo stratum). If contamination drove these
     numbers, they should move when the candidates are removed.
  3. Direct check: are the candidate controls judged "AI" more often than the
     pre-2023 controls (Fisher exact test)?

The audio-based check promised in the rebuttal (running the released feature
detector over the controls) needs the preview audio and detector weights and is
not part of this script.

Data convention: v2 export only (it already contains all v1 annotations).

Outputs
  output/control_set_integrity.csv
  output/control_set_integrity_table.tex

Run:  .venv/bin/python control_set_integrity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"

CUTOFF_YEAR = 2023  # first year any generator in this study was publicly usable
RATING = "aesthetic_quality"
N_BOOT = 10_000
SEED = 42


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ann = pd.DataFrame(json.loads((INPUT / "v2" / "annotations_export_2026-04-20.json").read_text()))
    assert len(ann) == 591, f"expected 591, got {len(ann)}"
    ann[RATING] = pd.to_numeric(ann[RATING], errors="coerce")
    ann["uuid"] = ann.song_id.str.split("_", n=1).str[1]
    pool = pd.DataFrame(
        # split on "\n" only: str.splitlines() also breaks on U+2028 inside JSON strings
        [json.loads(l) for l in (INPUT / "annotation-subset" / "human_songs.jsonl").read_text().split("\n") if l.strip()]
    )
    return ann, pool


def boot_diff(a, b, n=N_BOOT, seed=SEED):
    """Bootstrap CI for mean(a) - mean(b) (same helper as halo_robustness.py)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, (np.nan, np.nan), len(a), len(b)
    d = a.mean() - b.mean()
    reps = np.empty(n)
    for i in range(n):
        reps[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return d, (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))), len(a), len(b)


def year_bucket(date) -> str:
    if not isinstance(date, str) or len(date) < 4:
        return "undated"
    return f"pre-{CUTOFF_YEAR}" if int(date[:4]) < CUTOFF_YEAR else f"{CUTOFF_YEAR}+"


def summarize(label: str, human: pd.DataFrame, hit_rate: float) -> dict:
    det = human[human.authenticity_assessment != "uncertain"]
    n_fa = int((det.authenticity_assessment == "ai-generated").sum())
    n_cr = int((det.authenticity_assessment == "real").sum())
    fa = n_fa / (n_fa + n_cr)
    # Same convention as the paper's pooled d' (aggregate rates, no correction).
    d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa)
    bacc = 0.5 * (hit_rate + (1 - fa))
    r = det.loc[det.authenticity_assessment == "real", RATING]
    a = det.loc[det.authenticity_assessment == "ai-generated", RATING]
    d, ci, _, _ = boot_diff(r, a)
    return {
        "subset": label,
        "n_songs": int(human.uuid.nunique()),
        "n_trials": int(len(human)),
        "n_non_uncertain": int(len(det)),
        "false_alarm_rate": round(fa, 3),
        "d_prime": round(float(d_prime), 2),
        "balanced_accuracy": round(float(bacc), 3),
        "mean_aesthetic_quality": round(float(human[RATING].mean()), 2),
        "halo_gap_within_human": round(float(d), 2) if not np.isnan(d) else np.nan,
        "halo_ci_low": round(ci[0], 2) if not np.isnan(d) else np.nan,
        "halo_ci_high": round(ci[1], 2) if not np.isnan(d) else np.nan,
    }


def main() -> None:
    ann, pool = load()
    pool["release_bucket"] = pool.album_release_date.map(year_bucket)
    human = ann[ann.song_source == "human"].merge(
        pool[["uuid", "album_release_date", "release_bucket", "language"]], on="uuid", how="left"
    )
    assert human.release_bucket.notna().all(), "every benchmark control must be in the pool"

    ai = ann[(ann.song_source != "human") & (ann.authenticity_assessment != "uncertain")]
    hit_rate = float((ai.authenticity_assessment == "ai-generated").mean())

    songs = human.drop_duplicates("uuid")
    counts = songs.release_bucket.value_counts()
    n_songs = len(songs)
    n_pre = int(counts.get(f"pre-{CUTOFF_YEAR}", 0))
    n_candidates = n_songs - n_pre
    years = pd.to_numeric(songs.album_release_date.str[:4], errors="coerce").dropna()
    print(f"Benchmark controls: {n_songs} songs, {len(human)} trials")
    print(f"Release buckets: {counts.to_dict()}")
    print(f"Release year min/median/max: {int(years.min())} / {int(years.median())} / {int(years.max())}")
    print(f"Contamination candidates (released {CUTOFF_YEAR}+ or undated): "
          f"{n_candidates}/{n_songs} = {n_candidates / n_songs:.1%}")
    print(f"Languages: {songs.language.value_counts().head(6).to_dict()}")

    pre = human[human.release_bucket == f"pre-{CUTOFF_YEAR}"]
    cand = human[human.release_bucket != f"pre-{CUTOFF_YEAR}"]
    rows = [
        summarize("All controls", human, hit_rate),
        summarize(f"Released before {CUTOFF_YEAR}", pre, hit_rate),
        summarize(f"Released {CUTOFF_YEAR}+ or undated", cand, hit_rate),
    ]

    # Are the candidates judged AI more often than the pre-cutoff controls?
    def ai_counts(df):
        d = df[df.authenticity_assessment != "uncertain"]
        k = int((d.authenticity_assessment == "ai-generated").sum())
        return k, len(d) - k
    (k1, r1), (k2, r2) = ai_counts(pre), ai_counts(cand)
    _, fisher_p = stats.fisher_exact([[k1, r1], [k2, r2]])
    print(f"Said-AI rate: pre {k1}/{k1 + r1}, candidates {k2}/{k2 + r2}; Fisher p = {fisher_p:.3f}")

    out = pd.DataFrame(rows)
    out["release_year_min"] = int(years.min())
    out["release_year_median"] = int(years.median())
    out["release_year_max"] = int(years.max())
    out["n_candidate_songs"] = n_candidates
    out["fisher_p_said_ai_candidates_vs_pre"] = round(float(fisher_p), 3)
    out["ai_hit_rate_fixed"] = round(hit_rate, 3)
    OUTPUT.mkdir(exist_ok=True)
    out.to_csv(OUTPUT / "control_set_integrity.csv", index=False)
    print("\n" + out[["subset", "n_songs", "n_trials", "false_alarm_rate", "d_prime",
                      "balanced_accuracy", "halo_gap_within_human", "halo_ci_low", "halo_ci_high"]].to_string(index=False))

    def ci(r):
        if pd.isna(r.halo_gap_within_human):
            return "--"
        return f"${r.halo_gap_within_human:+.2f}$ $[{r.halo_ci_low:+.2f}, {r.halo_ci_high:+.2f}]$"

    lines = [
        "% Generated by analysis/control_set_integrity.py -- do not edit manually",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"\textbf{Human controls} & \textbf{Songs} & \textbf{Trials} & \textbf{FA} & $d'$ & \textbf{BAcc} & \textbf{Halo $\Delta$ within human} \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        lines.append(f"{r.subset} & {r.n_songs} & {r.n_trials} & {r.false_alarm_rate:.2f} & "
                     f"{r.d_prime:.2f} & {r.balanced_accuracy:.2f} & {ci(r)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "control_set_integrity_table.tex").write_text("\n".join(lines) + "\n")
    print("\nWrote output/control_set_integrity.csv and output/control_set_integrity_table.tex")


if __name__ == "__main__":
    main()
