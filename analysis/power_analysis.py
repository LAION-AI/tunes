"""
A7 — Power / precision analysis for the benchmark's headline claims.

Addresses reviewer Uhv7-W3: "reporting power analyses and being explicit that
detection rates are sample-specific would be appropriate."

Methodological note: post-hoc power computed *at the observed effect size* is
circular (it is a monotone restatement of the p-value). We therefore lead with
the **MDES** (minimum detectable effect size): given the realized sample sizes,
alpha = 0.05, and 80% power, what is the smallest effect this design could have
caught? That is a property of the design, not of the result. Observed-effect
power is reported alongside only as a descriptive companion, and CI widths are
given because for a benchmark the honest question is precision, not power.

Claims audited
  C1  Per-participant d' > 0 (the paper's t(9) = 3.09 test)
  C2  Halo: judged-real minus judged-AI aesthetic gap
  C3  Source contrasts: human vs pooled AI on the five rating dimensions
  C4  Listener-subgroup detection differences (professional vs casual)

Data convention: v2 export only (it already contains all v1 annotations).

Outputs
  output/power_analysis.csv
  output/power_analysis_table.tex

Run:  .venv/bin/python power_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

BASE = Path(__file__).parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"

ALPHA = 0.05
TARGET_POWER = 0.80
RATING_COLS = [
    "aesthetic_quality", "playlist_likelihood", "musical_creativity",
    "production_quality", "emotional_engagement",
]


def load() -> pd.DataFrame:
    v2 = json.loads((INPUT / "v2" / "annotations_export_2026-04-20.json").read_text())
    df = pd.DataFrame(v2)
    assert len(df) == 591, f"expected 591, got {len(df)}"
    for c in RATING_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_ai_song"] = df.song_source != "human"
    df["detected_ai"] = (df.authenticity_assessment == "ai-generated").astype(int)
    return df


def df_row_power(rows: list[dict], claim: str) -> float:
    """Achieved power for a recorded claim (helper for computed commentary)."""
    for r in rows:
        if r.get("claim") == claim:
            return float(r.get("achieved_power", float("nan")))
    return float("nan")


def sdt_d_prime(hit, miss, fa, cr):
    """Log-linear (Hautus) corrected d'."""
    hr = (hit + 0.5) / (hit + miss + 1)
    far = (fa + 0.5) / (fa + cr + 1)
    return stats.norm.ppf(hr) - stats.norm.ppf(far)


def per_participant_dprime(ann: pd.DataFrame) -> pd.Series:
    """d' for participants having BOTH signal (AI) and noise (human) trials."""
    out = {}
    for pid, g in ann.groupby("participant_id"):
        d = g[g.authenticity_assessment != "uncertain"]
        if d.is_ai_song.sum() == 0 or (~d.is_ai_song).sum() == 0:
            continue
        out[pid] = sdt_d_prime(
            int(((d.is_ai_song) & (d.detected_ai == 1)).sum()),
            int(((d.is_ai_song) & (d.detected_ai == 0)).sum()),
            int(((~d.is_ai_song) & (d.detected_ai == 1)).sum()),
            int(((~d.is_ai_song) & (d.detected_ai == 0)).sum()),
        )
    return pd.Series(out, dtype=float)


def main() -> None:
    ann = load()
    rows = []

    one = TTestPower()
    two = TTestIndPower()

    # ------------------------------------------------------------------ C1
    print("=" * 78)
    print("C1  Per-participant d' > 0  (the paper's one-sample t-test)")
    print("=" * 78)
    dp = per_participant_dprime(ann).dropna()
    n = len(dp)
    t, p = stats.ttest_1samp(dp, 0.0)
    d_eff = dp.mean() / dp.std(ddof=1)
    ci = stats.t.interval(0.95, n - 1, loc=dp.mean(), scale=stats.sem(dp))
    mdes = one.solve_power(effect_size=None, nobs=n, alpha=ALPHA, power=TARGET_POWER,
                           alternative="two-sided")
    ach = one.power(effect_size=d_eff, nobs=n, alpha=ALPHA, alternative="two-sided")
    print(f"  n participants with both AI and human trials = {n}  (of {ann.participant_id.nunique()} total)")
    print(f"  mean d' = {dp.mean():.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  sd={dp.std(ddof=1):.3f}")
    print(f"  t({n-1}) = {t:.3f}, p = {p:.4f},  Cohen's d_z = {d_eff:.3f}")
    print(f"  MDES at 80% power = d_z {mdes:.3f}  (i.e. d' of {mdes*dp.std(ddof=1):.2f} on this scale)")
    print(f"  achieved power at observed effect = {ach:.3f}")
    print("  -> Well powered for the *existence* of above-chance detection, but n is")
    print("     small: only participants who saw both AI and human songs qualify.")
    rows.append({
        "claim": "C1_per_participant_dprime_gt_0", "n": n,
        "estimate": round(float(dp.mean()), 3),
        "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3),
        "effect_size_dz": round(float(d_eff), 3), "p": round(float(p), 4),
        "mdes_80pct": round(float(mdes), 3),
        "achieved_power": round(float(ach), 3),
    })

    # ------------------------------------------------------------------ C2
    print("\n" + "=" * 78)
    print("C2  Halo: judged-real minus judged-AI aesthetic-quality gap")
    print("=" * 78)
    det = ann[ann.authenticity_assessment != "uncertain"]
    a = det.loc[det.authenticity_assessment == "real", "aesthetic_quality"].dropna()
    b = det.loc[det.authenticity_assessment == "ai-generated", "aesthetic_quality"].dropna()
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d_eff = (a.mean() - b.mean()) / pooled
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    diff = a.mean() - b.mean()
    mdes = two.solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=ALPHA,
                           power=TARGET_POWER, alternative="two-sided")
    ach = two.power(effect_size=d_eff, nobs1=n1, ratio=n2 / n1, alpha=ALPHA,
                    alternative="two-sided")
    print(f"  n = {n1} judged-real vs {n2} judged-AI trials")
    print(f"  gap = {diff:+.3f} points, 95% CI [{diff-1.96*se:+.3f}, {diff+1.96*se:+.3f}]")
    print(f"  Cohen's d = {d_eff:.3f}, Welch p = {p2:.3e}")
    print(f"  MDES at 80% power = d {mdes:.3f} (~{mdes*pooled:.2f} rating points)")
    print(f"  achieved power at observed effect = {ach:.4f}")
    print(f"  -> Very well powered; the design could have detected an effect "
          f"{abs(d_eff)/mdes:.1f}x smaller than the one observed.")
    rows.append({
        "claim": "C2_halo_gap", "n": n1 + n2,
        "estimate": round(float(diff), 3),
        "ci_low": round(float(diff - 1.96 * se), 3), "ci_high": round(float(diff + 1.96 * se), 3),
        "effect_size_dz": round(float(d_eff), 3), "p": float(p2),
        "mdes_80pct": round(float(mdes), 3), "achieved_power": round(float(ach), 4),
    })

    # ------------------------------------------------------------------ C3
    print("\n" + "=" * 78)
    print("C3  Source contrasts: human vs pooled AI, per rating dimension")
    print("=" * 78)
    print(f"  {'dimension':24s} {'d':>7s} {'p':>9s} {'MDES':>7s} {'power':>7s}")
    for col in RATING_COLS:
        h = ann.loc[~ann.is_ai_song, col].dropna()
        ai = ann.loc[ann.is_ai_song, col].dropna()
        n1, n2 = len(h), len(ai)
        pooled = np.sqrt(((n1 - 1) * h.var(ddof=1) + (n2 - 1) * ai.var(ddof=1)) / (n1 + n2 - 2))
        d_eff = (h.mean() - ai.mean()) / pooled
        _, pv = stats.mannwhitneyu(h, ai, alternative="two-sided")
        mdes = two.solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=ALPHA,
                               power=TARGET_POWER, alternative="two-sided")
        ach = two.power(effect_size=abs(d_eff), nobs1=n1, ratio=n2 / n1, alpha=ALPHA,
                        alternative="two-sided")
        flag = "" if ach >= TARGET_POWER else "  <- underpowered"
        print(f"  {col:24s} {d_eff:7.3f} {pv:9.3f} {mdes:7.3f} {ach:7.3f}{flag}")
        rows.append({
            "claim": f"C3_source_contrast_{col}", "n": n1 + n2,
            "estimate": round(float(h.mean() - ai.mean()), 3),
            "effect_size_dz": round(float(d_eff), 3), "p": float(pv),
            "mdes_80pct": round(float(mdes), 3), "achieved_power": round(float(ach), 3),
        })
    n_h, n_ai = int((~ann.is_ai_song).sum()), int(ann.is_ai_song.sum())
    n_under = sum(1 for c in RATING_COLS
                  if df_row_power(rows, f"C3_source_contrast_{c}") < TARGET_POWER)
    print(f"  (n_human = {n_h} vs n_AI = {n_ai}; human share {n_h/(n_h+n_ai)*100:.0f}%)")
    print(f"  -> The small human share is the binding constraint: only effects of")
    print(f"     d >~ {mdes:.2f} are reliably detectable. {n_under} of {len(RATING_COLS)} "
          f"dimensions fall below {TARGET_POWER:.0%} power, so null AI-vs-human")
    print("     contrasts mean 'not detectably different', NOT 'equivalent'.")

    # ------------------------------------------------------------------ C4
    print("\n" + "=" * 78)
    print("C4  Subgroup detection accuracy (professional vs casual)")
    print("=" * 78)
    d = ann[ann.authenticity_assessment != "uncertain"].copy()
    d["correct"] = ((d.detected_ai == 1) & d.is_ai_song) | ((d.detected_ai == 0) & ~d.is_ai_song)
    grp = d.groupby("participant_musical_engagement").correct.agg(["mean", "size"])
    print(grp.round(3).to_string())
    for lvl in grp.index:
        k = int(d.loc[d.participant_musical_engagement == lvl, "correct"].sum())
        nn = int(grp.loc[lvl, "size"])
        lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, nn - k + 0.5)  # Jeffreys
        half = (hi - lo) / 2
        print(f"  {str(lvl):28s} acc={k/nn:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (+/-{half:.3f}, n={nn})")
        rows.append({
            "claim": f"C4_subgroup_accuracy_{lvl}", "n": nn,
            "estimate": round(k / nn, 3), "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3), "ci_halfwidth": round(float(half), 3),
        })
    print("  -> Subgroup accuracies carry wide intervals; treat as sample-specific")
    print("     descriptives, not population estimates.")

    # -------------------------------------------------------------- outputs
    df = pd.DataFrame(rows)
    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / "power_analysis.csv", index=False)
    print(f"\nWrote output/power_analysis.csv")

    def fp(p):
        if p >= 1e-4:
            return f"{p:.3f}"
        e = int(np.floor(np.log10(p)))
        return f"{p/10**e:.1f}\\times 10^{{{e}}}"

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Claim} & \textbf{$n$} & \textbf{Effect} & \textbf{$p$} & \textbf{MDES}"
        r" & \textbf{Power} \\",
        r"\midrule",
    ]
    pretty = {
        "C1_per_participant_dprime_gt_0": r"Per-participant $d' > 0$",
        "C2_halo_gap": r"Halo gap (judged real $-$ AI)",
    }
    for key in ("C1_per_participant_dprime_gt_0", "C2_halo_gap"):
        r = df[df.claim == key].iloc[0]
        lines.append(f"{pretty[key]} & {int(r.n)} & ${r.effect_size_dz:.2f}$ & "
                     f"${fp(r.p)}$ & ${r.mdes_80pct:.2f}$ & ${r.achieved_power:.2f}$ \\\\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{6}{@{}l}{\emph{Human vs.\ pooled-AI rating contrasts:}} \\")
    for col in RATING_COLS:
        r = df[df.claim == f"C3_source_contrast_{col}"].iloc[0]
        nm = col.replace("_", " ").capitalize()
        lines.append(f"\\quad {nm} & {int(r.n)} & ${r.effect_size_dz:.2f}$ & "
                     f"${fp(r.p)}$ & ${r.mdes_80pct:.2f}$ & ${r.achieved_power:.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "power_analysis_table.tex").write_text("\n".join(lines) + "\n")
    print("Wrote output/power_analysis_table.tex")


if __name__ == "__main__":
    main()
