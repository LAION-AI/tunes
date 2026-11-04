"""
A3 — Representativeness of the 572-song benchmark subset vs. the 1.4M corpus.

Addresses reviewer N3D6-W2 ("Why only 572 of 1.4M, and how is this slice
representative? Does platform imbalance bias the results?") and Uhv7-W3.

Three nested populations are compared:

  corpus  : the full 1,429,734-track REDACTED-Tunes release. Per-platform
            reference statistics are taken from the manuscript's Table
            `tab:redactedtunes_perplatform` / App. `app:dataset_extra`,
            which were computed over the full corpus. The per-track corpus
            parquets are not in this repo, so these enter as documented
            constants (see CORPUS_REFERENCE).
  pool    : the candidate sampling frame actually drawn from the corpus
            (input/annotation-subset/*.jsonl, 2,500 tracks per AI platform).
  bench   : the 572 unique songs that received >=1 human annotation.

We test whether `bench` is a faithful draw of `pool` (two-sample KS +
Mann-Whitney on duration; chi-square on genre), and whether `pool`/`bench`
track the published `corpus` percentiles.

Outputs
  output/subset_representativeness.csv        (tidy, machine-readable)
  output/subset_representativeness_table.tex  (\input-able LaTeX table)

Run:  .venv/bin/python subset_representativeness.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"
POOL_DIR = INPUT / "annotation-subset"

RNG_NOTE = "no randomness: all statistics are deterministic given the inputs"

# --- Corpus reference values -------------------------------------------------
# Source: manuscript Table `tab:redactedtunes_perplatform` and App.
# `app:dataset_extra`, computed over the full 1,429,734-track release.
# Durations in seconds; aesthetics on the 1-5 SongEval-calibrated scale.
CORPUS_REFERENCE = {
    "suno":   {"tracks": 1_037_381, "median": 185.4, "p10": 118.6, "p90": 241.4, "aesthetics": 3.28},
    "mureka": {"tracks":   383_549, "median": 138.2, "p10":  48.1, "p90": 194.7, "aesthetics": 3.34},
    "udio":   {"tracks":     8_804, "median": 131.0, "p10":  32.8, "p90": 261.0, "aesthetics": 3.08},
}
CORPUS_TOTAL = sum(v["tracks"] for v in CORPUS_REFERENCE.values())
CORPUS_GLOBAL_AESTHETICS = 3.29  # App. `app:dataset_extra`

# Platforms that are actually part of the released corpus. Sonauto and the
# Apple Music human controls are benchmark-only stimuli and are therefore
# excluded from corpus-representativeness claims (but reported separately).
RELEASE_PLATFORMS = ["suno", "mureka", "udio"]

# Duration field + unit per source file. Verified empirically:
# mureka/suno annotation exports are milliseconds, udio is seconds.
POOL_SPEC = {
    "suno":   ("suno_subset.jsonl",   lambda d: d.get("metadata", {}).get("duration"), 1.0),
    "mureka": ("mureka_subset.jsonl", lambda d: d.get("duration in milliseconds"),     1e-3),
    "udio":   ("udio_subset.jsonl",   lambda d: d.get("duration"),                     1.0),
}
BENCH_DURATION_SCALE = {"suno": 1e-3, "mureka": 1e-3, "udio": 1.0}


def load_benchmark_songs() -> pd.DataFrame:
    """Unique songs across v1+v2 annotations (one row per song)."""
    v1 = json.loads((INPUT / "v1" / "annotations_export_2026-04-07.json").read_text())
    v2 = json.loads((INPUT / "v2" / "annotations_export_2026-04-20.json").read_text())
    songs = {}
    for a in v1 + v2:
        songs.setdefault(a["song_id"], a)
    rows = []
    for sid, a in songs.items():
        src = a.get("song_source")
        raw = a.get("song_duration")
        dur = None
        if raw not in (None, "", 0):
            dur = float(raw) * BENCH_DURATION_SCALE.get(src, 1.0)
        rows.append({
            "song_id": sid,
            "source": src,
            "duration_s": dur,
            "genres": a.get("song_genres"),
            "tags": a.get("song_tags"),
        })
    return pd.DataFrame(rows)


def load_pool() -> pd.DataFrame:
    rows = []
    for src, (fname, getter, scale) in POOL_SPEC.items():
        for line in (POOL_DIR / fname).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = getter(d)
            rows.append({
                "source": src,
                "duration_s": float(v) * scale if v else None,
                "genres": d.get("genres"),
                "tags": d.get("tags"),
            })
    return pd.DataFrame(rows)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    bench = load_benchmark_songs()
    pool = load_pool()

    print(f"Benchmark songs: {len(bench)}  |  pool tracks: {len(pool)}")
    print(bench["source"].value_counts().to_string(), "\n")

    records = []

    # ---------------------------------------------------------------- duration
    print("=" * 78)
    print("DURATION (seconds) — benchmark vs pool vs corpus")
    print("=" * 78)
    for src in RELEASE_PLATFORMS:
        b = bench.loc[(bench.source == src) & bench.duration_s.notna(), "duration_s"].values
        p = pool.loc[(pool.source == src) & pool.duration_s.notna(), "duration_s"].values
        ref = CORPUS_REFERENCE[src]

        ks_s, ks_p = stats.ks_2samp(b, p)
        mw_s, mw_p = stats.mannwhitneyu(b, p, alternative="two-sided")
        # Rank-biserial correlation as a scale-free effect size for MWU.
        rbc = 1 - (2 * mw_s) / (len(b) * len(p))

        print(f"\n{src.upper()}  (bench n={len(b)}, pool n={len(p)}, corpus n={ref['tracks']:,})")
        print(f"  median  bench={np.median(b):7.1f}  pool={np.median(p):7.1f}  corpus={ref['median']:7.1f}"
              f"   (bench-corpus {np.median(b)-ref['median']:+.1f}s)")
        print(f"  P10     bench={pct(b,10):7.1f}  pool={pct(p,10):7.1f}  corpus={ref['p10']:7.1f}")
        print(f"  P90     bench={pct(b,90):7.1f}  pool={pct(p,90):7.1f}  corpus={ref['p90']:7.1f}")
        print(f"  bench vs pool: KS D={ks_s:.3f} p={ks_p:.3f} | MWU p={mw_p:.3f} rank-biserial={rbc:+.3f}")

        records.append({
            "comparison": "duration_s", "platform": src,
            "bench_n": len(b), "pool_n": len(p), "corpus_n": ref["tracks"],
            "bench_median": round(float(np.median(b)), 1),
            "pool_median": round(float(np.median(p)), 1),
            "corpus_median": ref["median"],
            "bench_p10": round(pct(b, 10), 1), "pool_p10": round(pct(p, 10), 1), "corpus_p10": ref["p10"],
            "bench_p90": round(pct(b, 90), 1), "pool_p90": round(pct(p, 90), 1), "corpus_p90": ref["p90"],
            "ks_D": round(float(ks_s), 4), "ks_p": round(float(ks_p), 4),
            "mwu_p": round(float(mw_p), 4), "rank_biserial": round(float(rbc), 4),
        })

    # -------------------------------------------------------------- aesthetics
    print("\n" + "=" * 78)
    print("AESTHETICS (1-5, our released head) — benchmark vs corpus mean")
    print("=" * 78)
    aes_path = OUTPUT / "our_music_aesthetics_song_scores.csv"
    if aes_path.exists():
        aes = pd.read_csv(aes_path)
        # The corpus aesthetics field is computed on a fixed 30s slice
        # (§sec:lt_pipeline), so `cropped` benchmark songs are the
        # apples-to-apples stratum; `full` songs are scored over the whole
        # track and read systematically differently. We report the pooled
        # comparison as primary and the cropped-only stratum as the
        # pipeline-matched sensitivity check.
        for stratum, sub in (("all", aes), ("cropped_only", aes[aes.snippet_kind == "cropped"])):
            print(f"\n  -- stratum: {stratum} --")
            for src in RELEASE_PLATFORMS:
                v = sub.loc[sub.song_source == src, "raw_Overall_Aesthetics"].dropna().values
                if len(v) < 3:
                    continue
                ref = CORPUS_REFERENCE[src]["aesthetics"]
                t, p_val = stats.ttest_1samp(v, ref)
                d = (v.mean() - ref) / v.std(ddof=1)  # Cohen's d vs corpus mean
                print(f"  {src:7s} bench mean={v.mean():.3f} (sd {v.std(ddof=1):.3f}, n={len(v):3d})"
                      f"  corpus={ref:.2f}  Δ={v.mean()-ref:+.3f}  t={t:+.2f} p={p_val:.3f} d={d:+.2f}")
                records.append({
                    "comparison": f"aesthetics_1to5_{stratum}", "platform": src,
                    "bench_n": len(v), "corpus_n": CORPUS_REFERENCE[src]["tracks"],
                    "bench_mean": round(float(v.mean()), 3), "bench_sd": round(float(v.std(ddof=1)), 3),
                    "corpus_mean": ref, "delta": round(float(v.mean() - ref), 3),
                    "t": round(float(t), 3), "p": round(float(p_val), 4), "cohens_d": round(float(d), 3),
                })
    else:
        print("  [skipped] aesthetics score file not found")

    # ------------------------------------------------------------ platform mix
    print("\n" + "=" * 78)
    print("PLATFORM MIX — benchmark vs corpus (deliberate rebalancing)")
    print("=" * 78)
    bench_rel = bench[bench.source.isin(RELEASE_PLATFORMS)]
    for src in RELEASE_PLATFORMS:
        b_share = (bench_rel.source == src).mean()
        c_share = CORPUS_REFERENCE[src]["tracks"] / CORPUS_TOTAL
        print(f"  {src:7s} benchmark {b_share*100:5.1f}%   corpus {c_share*100:5.1f}%   "
              f"({'over' if b_share > c_share else 'under'}-sampled {b_share/c_share:.1f}x)")
        records.append({
            "comparison": "platform_share", "platform": src,
            "bench_n": int((bench_rel.source == src).sum()),
            "bench_share_pct": round(float(b_share * 100), 1),
            "corpus_share_pct": round(float(c_share * 100), 2),
            "ratio": round(float(b_share / c_share), 2),
        })

    # -------------------------------------------------------------------- genre
    print("\n" + "=" * 78)
    print("GENRE (Mureka — only platform with a clean controlled genre vocabulary)")
    print("=" * 78)

    def parse_genres(x):
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except json.JSONDecodeError:
                return []
        return [g.strip().lower() for g in x if isinstance(g, str) and g.strip()] if isinstance(x, list) else []

    b_g = Counter(g for x in bench.loc[bench.source == "mureka", "genres"] for g in parse_genres(x))
    p_g = Counter(g for x in pool.loc[pool.source == "mureka", "genres"] for g in parse_genres(x))
    if b_g and p_g:
        keys = [k for k, _ in p_g.most_common(10)]
        b_tot, p_tot = sum(b_g.values()), sum(p_g.values())
        obs = np.array([b_g.get(k, 0) for k in keys], dtype=float)
        exp_share = np.array([p_g[k] / p_tot for k in keys])
        exp = exp_share / exp_share.sum() * obs.sum()
        chi2, chi_p = stats.chisquare(obs, exp)
        print(f"  top-10 pool genres, benchmark vs pool share:  chi2={chi2:.2f} p={chi_p:.3f}")
        for k in keys:
            print(f"    {k:16s} bench {b_g.get(k,0)/b_tot*100:5.1f}%   pool {p_g[k]/p_tot*100:5.1f}%")
        records.append({
            "comparison": "genre_mureka_top10", "platform": "mureka",
            "bench_n": int(obs.sum()), "pool_n": int(p_tot),
            "chi2": round(float(chi2), 3), "p": round(float(chi_p), 4),
        })

    # ------------------------------------------------------------------ outputs
    df = pd.DataFrame(records)
    OUTPUT.mkdir(exist_ok=True)
    csv_path = OUTPUT / "subset_representativeness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path.relative_to(BASE)}")

    # LaTeX table: duration + aesthetics, the two corpus-comparable axes.
    dur = df[df.comparison == "duration_s"].set_index("platform")
    aes_rows = df[df.comparison == "aesthetics_1to5_all"].set_index("platform") if "aesthetics_1to5_all" in set(df.comparison) else None
    share = df[df.comparison == "platform_share"].set_index("platform")

    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Duration median (s)}} & \multicolumn{2}{c}{\textbf{Aesthetics (1--5)}}"
        r" & \multicolumn{2}{c}{\textbf{Share (\%)}} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"\textbf{Platform} & \textbf{Bench.} & \textbf{Pool} & \textbf{Corpus} & \textbf{Bench.} & \textbf{Corpus}"
        r" & \textbf{Bench.} & \textbf{Corpus} & \textbf{KS $p$} \\",
        r"\midrule",
    ]
    for src in RELEASE_PLATFORMS:
        d = dur.loc[src]
        a_b = f"{aes_rows.loc[src,'bench_mean']:.2f}" if aes_rows is not None and src in aes_rows.index else "--"
        a_c = f"{aes_rows.loc[src,'corpus_mean']:.2f}" if aes_rows is not None and src in aes_rows.index else "--"
        lines.append(
            f"{src.capitalize()} & {d.bench_median:.1f} & {d.pool_median:.1f} & {d.corpus_median:.1f} & "
            f"{a_b} & {a_c} & {share.loc[src,'bench_share_pct']:.1f} & "
            f"{share.loc[src,'corpus_share_pct']:.1f} & {d.ks_p:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = OUTPUT / "subset_representativeness_table.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {tex_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
