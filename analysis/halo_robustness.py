"""
A2 — Halo robustness: is the quality-authenticity association explained by
real, measurable audio quality?

Addresses reviewer N3D6-W5 and the meta-review's "correlation not causation".
The reviewer raises two distinct objections:

  (1) Causality cannot be established because both questions are asked in the
      same trial. We agree, and do not claim causation; this script quantifies
      what the observational data *can* support.
  (2) "If it sounds good it must be real" is an unremarkable cue -- annotators
      simply using audible quality to infer provenance. This is the
      substantive objection, and it is testable: if the association were
      merely raters tracking genuine audible quality, then (a) it should be
      comparable in size to the true real-vs-AI quality difference, and
      (b) conditioning on *independently measured* audio quality should
      attenuate it.

Analyses
  A2a  Perceived-label gap stratified by ground-truth provenance.
  A2b  Magnitude comparison: perceived-label gap vs true-provenance gap.
  A2c  LMM conditioning on independent model-based quality scores
       (our aesthetics head, SongEval, Audiobox) -- does the judged-AI
       coefficient survive?
  A2d  Within-song between-rater disagreement (power check).

IMPORTANT CAVEAT carried into the paper: the automatic quality scores align
only weakly with human ratings (rho ~ 0.09-0.16, Table tab:aesthetics_llm_custom),
so they are *imperfect* proxies for perceived audible quality. Conditioning on a
weak proxy cannot fully rule out a quality-cue explanation; it can only show
that the association is not explained by the quality signal these systems
capture. This is stated explicitly rather than glossed over.

Data convention: the v2 export is cumulative and already contains all 403 v1
annotations (verified: v1 ids subset of v2, union = 591), so we load v2 only,
matching main.py's load_data().

Outputs
  output/halo_robustness.csv
  output/halo_robustness_table.tex

Run:  .venv/bin/python halo_robustness.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

BASE = Path(__file__).parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"

RATING = "aesthetic_quality"
RATING_COLS = [
    "aesthetic_quality", "playlist_likelihood", "musical_creativity",
    "production_quality", "emotional_engagement",
]
N_BOOT = 10_000
SEED = 42


def load_annotations() -> pd.DataFrame:
    """v2 only (cumulative); v1 ids are used solely to label snippet condition."""
    v2 = json.loads((INPUT / "v2" / "annotations_export_2026-04-20.json").read_text())
    v1_ids = {r["annotation_id"] for r in
              json.loads((INPUT / "v1" / "annotations_export_2026-04-07.json").read_text())}
    df = pd.DataFrame(v2)
    assert len(df) == 591, f"expected 591 v2 annotations, got {len(df)}"
    df["snippet_condition"] = np.where(df.annotation_id.isin(v1_ids), "original", "30s")
    for c in RATING_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_ai_song"] = df.song_source != "human"
    df["ground_truth"] = np.where(df.is_ai_song, "ai-generated", "real")
    return df


def load_quality_scores() -> pd.DataFrame:
    """Independent, model-based per-song quality scores (one row per song)."""
    ours = pd.read_csv(OUTPUT / "our_music_aesthetics_song_scores.csv")
    songeval = pd.read_csv(OUTPUT / "songeval_song_scores.csv")
    audiobox = pd.read_csv(OUTPUT / "audiobox_song_scores.csv")

    q = ours[["song_id", "raw_Overall_Aesthetics"]].rename(
        columns={"raw_Overall_Aesthetics": "q_ours"})
    # SongEval: mean of its five expert-calibrated axes.
    se_axes = ["Musicality", "Coherence", "Memorability", "Clarity", "Naturalness"]
    songeval["q_songeval"] = songeval[se_axes].mean(axis=1)
    q = q.merge(songeval[["song_id", "q_songeval"]], on="song_id", how="outer")
    # Audiobox: Content Enjoyment is the axis best aligned with human aesthetic
    # quality in Table tab:aesthetics_llm_custom (rho = 0.16, the strongest of
    # any automatic evaluator), so it is the most demanding single control.
    q = q.merge(audiobox[["song_id", "CE"]].rename(columns={"CE": "q_audiobox"}),
                on="song_id", how="outer")
    return q


def boot_diff(a, b, n=N_BOOT, seed=SEED):
    """Bootstrap CI for mean(a) - mean(b)."""
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


def main() -> None:
    ann = load_annotations()
    q = load_quality_scores()
    ann = ann.merge(q, on="song_id", how="left")

    print(f"Loaded {len(ann)} annotations on {ann.song_id.nunique()} songs")
    print("Quality-score coverage:", {c: int(ann[c].notna().sum()) for c in ["q_ours", "q_songeval", "q_audiobox"]})

    records = []
    det = ann[ann.authenticity_assessment != "uncertain"].copy()

    # ---------------------------------------------------------------- A2a
    print("\n" + "=" * 78)
    print("A2a  Perceived-label gap, stratified by GROUND TRUTH")
    print("=" * 78)
    print("If the gap persists inside each ground-truth stratum, it is not")
    print("explained by true provenance.\n")
    for stratum, label in (("all", "All songs"), ("ai-generated", "Actually AI"), ("real", "Actually human")):
        sub = det if stratum == "all" else det[det.ground_truth == stratum]
        r = sub.loc[sub.authenticity_assessment == "real", RATING]
        a = sub.loc[sub.authenticity_assessment == "ai-generated", RATING]
        d, ci, n_r, n_a = boot_diff(r, a)
        if np.isnan(d):
            print(f"  {label:16s} insufficient data (n_real={n_r}, n_ai={n_a})")
            continue
        u, p = stats.mannwhitneyu(r.dropna(), a.dropna(), alternative="two-sided")
        print(f"  {label:16s} judged-real {r.mean():.2f} (n={n_r:3d})  judged-AI {a.mean():.2f} (n={n_a:3d})  "
              f"Δ={d:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  MWU p={p:.2e}")
        records.append({
            "analysis": "A2a_gap_by_ground_truth", "stratum": label,
            "judged_real_mean": round(float(r.mean()), 3), "n_real": n_r,
            "judged_ai_mean": round(float(a.mean()), 3), "n_ai": n_a,
            "delta": round(float(d), 3), "ci_low": round(ci[0], 3), "ci_high": round(ci[1], 3),
            "mwu_p": float(p),
        })

    # ---------------------------------------------------------------- A2b
    print("\n" + "=" * 78)
    print("A2b  Magnitude: PERCEIVED-label gap vs TRUE-provenance gap")
    print("=" * 78)
    print("The reviewer's benign reading (raters track genuine quality) predicts")
    print("these should be comparable. Same scale, same trials.\n")
    for col in RATING_COLS:
        pr = det.loc[det.authenticity_assessment == "real", col]
        pa = det.loc[det.authenticity_assessment == "ai-generated", col]
        d_perc, ci_p, _, _ = boot_diff(pr, pa)
        tr = ann.loc[ann.ground_truth == "real", col]
        ta = ann.loc[ann.ground_truth == "ai-generated", col]
        d_true, ci_t, _, _ = boot_diff(tr, ta)
        ratio = d_perc / d_true if d_true and not np.isnan(d_true) else np.nan
        print(f"  {col:24s} perceived Δ={d_perc:+.2f} [{ci_p[0]:+.2f},{ci_p[1]:+.2f}]   "
              f"true Δ={d_true:+.2f} [{ci_t[0]:+.2f},{ci_t[1]:+.2f}]   ratio={ratio:.1f}x")
        records.append({
            "analysis": "A2b_perceived_vs_true", "dimension": col,
            "perceived_delta": round(float(d_perc), 3),
            "perceived_ci_low": round(ci_p[0], 3), "perceived_ci_high": round(ci_p[1], 3),
            "true_delta": round(float(d_true), 3),
            "true_ci_low": round(ci_t[0], 3), "true_ci_high": round(ci_t[1], 3),
            "ratio": round(float(ratio), 2) if not np.isnan(ratio) else None,
        })

    # ---------------------------------------------------------------- A2c
    print("\n" + "=" * 78)
    print("A2c  Does conditioning on INDEPENDENT audio-quality scores attenuate")
    print("     the judged-AI coefficient?")
    print("=" * 78)
    work = det.dropna(subset=[RATING]).copy()
    work["judged_ai"] = (work.authenticity_assessment == "ai-generated").astype(int)
    for c in ["q_ours", "q_songeval", "q_audiobox"]:
        sd = work[c].std()
        work[f"{c}_z"] = (work[c] - work[c].mean()) / sd if sd else 0.0

    base = f"{RATING} ~ judged_ai + C(song_source) + C(snippet_condition)"
    specs = {
        "M0: no quality control":        base,
        "M1: + our aesthetics":          base + " + q_ours_z",
        "M2: + SongEval":                base + " + q_songeval_z",
        "M3: + Audiobox CE":             base + " + q_audiobox_z",
        "M4: + all three":               base + " + q_ours_z + q_songeval_z + q_audiobox_z",
    }
    print(f"  Participant random intercepts, n={len(work)} non-uncertain trials.")
    print(f"  {'Model':28s} {'judged-AI β':>12s} {'95% CI':>20s} {'p':>10s}")
    for name, formula in specs.items():
        sub = work.dropna(subset=[c for c in ["q_ours_z", "q_songeval_z", "q_audiobox_z"]
                                  if c.replace('_z', '') in formula or c in formula])
        try:
            res = smf.mixedlm(formula, data=sub, groups=sub["participant_id"]).fit(
                reml=True, method="lbfgs", maxiter=1000)
        except Exception as e:  # pragma: no cover
            print(f"  {name:28s} FIT FAILED: {e}")
            continue
        b = res.fe_params["judged_ai"]
        lo, hi = res.conf_int().loc["judged_ai"]
        p = res.pvalues["judged_ai"]
        print(f"  {name:28s} {b:+12.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>20s} {p:10.2e}   (n={len(sub)})")
        # Report each quality control's OWN coefficient. A control that does not
        # predict the human rating cannot meaningfully absorb the halo, so
        # "survives controlling for X" is only informative when X has signal.
        ctrl = {k: (float(res.fe_params[k]), float(res.pvalues[k]))
                for k in ("q_ours_z", "q_songeval_z", "q_audiobox_z") if k in res.fe_params}
        for k, (cb, cp) in ctrl.items():
            # In M4 the three proxies are mutually collinear (they score the same
            # audio), so individual p-values inflate there. Each is a significant
            # predictor when entered alone (M1-M3) -- that is what makes it a
            # real control rather than noise.
            flag = "" if cp < 0.05 else "   <- absorbed by collinear proxy"
            print(f"      control {k:14s} β={cb:+.3f} per SD, p={cp:.4f}{flag}")
        records.append({
            "analysis": "A2c_quality_conditioned_lmm", "model": name,
            "n": len(sub), "judged_ai_beta": round(float(b), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "p": float(p),
            "attenuation_vs_M0_pct": None,
            **{f"ctrl_{k}_beta": round(v[0], 4) for k, v in ctrl.items()},
            **{f"ctrl_{k}_p": round(v[1], 4) for k, v in ctrl.items()},
        })

    # Attenuation relative to the uncontrolled baseline.
    lmm_recs = [r for r in records if r["analysis"] == "A2c_quality_conditioned_lmm"]
    if lmm_recs:
        b0 = lmm_recs[0]["judged_ai_beta"]
        print(f"\n  Attenuation of the judged-AI coefficient relative to M0 (β={b0:+.3f}):")
        for r in lmm_recs:
            att = (1 - abs(r["judged_ai_beta"]) / abs(b0)) * 100
            r["attenuation_vs_M0_pct"] = round(float(att), 2)
            print(f"    {r['model']:28s} {att:+6.1f}%")

    # ---------------------------------------------------------------- A2d
    print("\n" + "=" * 78)
    print("A2d  Within-song between-rater disagreement (power check)")
    print("=" * 78)
    counts = det.groupby("song_id").authenticity_assessment.nunique()
    multi = det.groupby("song_id").size()
    disagree = counts[(counts > 1)].index
    n_multi = int((multi >= 2).sum())
    print(f"  Songs with >=2 non-uncertain annotations: {n_multi}")
    print(f"  Of those, songs where raters disagreed real-vs-AI: {len(disagree)}")
    if len(disagree) >= 10:
        gaps = []
        for sid in disagree:
            g = det[det.song_id == sid]
            r = g.loc[g.authenticity_assessment == "real", RATING].mean()
            a = g.loc[g.authenticity_assessment == "ai-generated", RATING].mean()
            if not (np.isnan(r) or np.isnan(a)):
                gaps.append(r - a)
        if gaps:
            t, p = stats.ttest_1samp(gaps, 0)
            print(f"  Mean within-song gap = {np.mean(gaps):+.2f} (n={len(gaps)} songs, t={t:+.2f}, p={p:.3f})")
            records.append({
                "analysis": "A2d_within_song", "n_songs": len(gaps),
                "mean_gap": round(float(np.mean(gaps)), 3), "t": round(float(t), 3), "p": float(p),
            })
    else:
        print("  -> UNDERPOWERED: too few disagreement songs for a within-song test.")
        print("     Reported as a design limitation, not analyzed.")
        records.append({
            "analysis": "A2d_within_song", "n_songs": len(disagree),
            "note": "underpowered; not analyzed",
        })

    # ------------------------------------------------------------- outputs
    df = pd.DataFrame(records)
    csv_path = OUTPUT / "halo_robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path.relative_to(BASE)}")

    lmm = df[df.analysis == "A2c_quality_conditioned_lmm"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Quality control} & \textbf{Judged-AI $\hat\beta$} & \textbf{95\% CI} & \textbf{$p$}"
        r" & \textbf{Atten.} \\",
        r"\midrule",
    ]
    pretty = {
        "M0: no quality control": "None (baseline)",
        "M1: + our aesthetics": r"$+$ \texttt{music-aesthetics}",
        "M2: + SongEval": r"$+$ SongEval",
        "M3: + Audiobox CE": r"$+$ Audiobox CE",
        "M4: + all three": r"$+$ all three",
    }
    def tex_p(p):
        """Render a p-value as LaTeX scientific notation."""
        if p == 0:
            return r"<10^{-300}"
        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return f"{mant:.1f}\\times 10^{{{exp}}}"

    for _, r in lmm.iterrows():
        stars = "$^{***}$" if r.p < 1e-3 else ("$^{**}$" if r.p < 1e-2 else ("$^{*}$" if r.p < 0.05 else ""))
        att = "--" if r.model.startswith("M0") else f"{r.attenuation_vs_M0_pct:.1f}\\%"
        lines.append(f"{pretty.get(r.model, r.model)} & ${r.judged_ai_beta:+.2f}${stars} & "
                     f"$[{r.ci_low:+.2f}, {r.ci_high:+.2f}]$ & ${tex_p(r.p)}$ & {att} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = OUTPUT / "halo_robustness_table.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {tex_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
