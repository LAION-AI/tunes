"""
Bootstrap confidence intervals for the caption benchmark.

Addresses reviewer zY8i: "For all three proposed models, their evaluations are
performed on roughly 100-200 data. It is really hard to determine whether the
model performance can generalize to a broader data distribution."

Two computations:

  (1) Paired bootstrap over clips on the non-Gemini averaged score from the
      de-circularization run (Voxtral-Small-24B and GPT-Audio-1.5, same 100
      Suno snippets). Paired, because every model is scored on the same clips,
      so the difference-of-means CI is the correct test rather than comparing
      two independent CIs.

  (2) Bootstrap on the mean per-snippet Accuracy of the original
      Gemini-3.1-Pro-judged benchmark, reconstructed from the published
      score-count distribution (Appendix Table `tab:score-dist`).

Result that matters: the difference between Music-Whisper (242M) and
MOSS-Audio-8B-Thinking (9.1B) is +0.02 with a 95% CI of [-0.16, +0.21]. The
two are statistically indistinguishable, so the claim "outperforms every open
model >=34x its size" is not supported for that comparison and must be stated
as a tie. Music-Whisper is significantly above MOSS-Audio-8B and Music-Flamingo,
both >=34x its size.

Outputs
  output/caption_eval_bootstrap.csv
  output/caption_eval_bootstrap_table.tex

Run:  .venv/bin/python caption_eval_bootstrap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
SRC = BASE / "input" / "caption_eval_decirc" / "caption_eval_decirc_first100.json"
JUDGES = ("voxtral", "gpt_audio")
SEED = 42
B = 20_000

# Per-snippet Accuracy score counts (scores 1..5) from Appendix Table
# `tab:score-dist` of the original Gemini-judged benchmark.
GEMINI_JUDGED_DIST = {
    "Music Whisper (ours, 242M)":     [6, 10, 8, 19, 56],
    "Gemini 2.5 Flash (closed)":      [3, 4, 3, 8, 81],
    "MOSS-Audio 8B-Thinking (9.1B)":  [2, 4, 6, 10, 77],
    "GPT-Audio 1.5 (closed)":         [11, 7, 6, 7, 62],
    "MOSS-Audio 8B (9.1B)":           [12, 27, 15, 29, 17],
    "Music Flamingo (8.3B)":          [13, 30, 18, 16, 21],
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = json.loads(SRC.read_text())
    models = d["models"]

    clips = None
    for m in models.values():
        for pj in m["per_judge"].values():
            s = set(pj["scores"])
            clips = s if clips is None else clips & s
    clips = sorted(clips)

    def per_clip(key: str) -> np.ndarray:
        pj = models[key]["per_judge"]
        return np.mean([[pj[j]["scores"][c] for c in clips] for j in JUDGES], axis=0)

    scores = {k: per_clip(k) for k in models}
    rows = []

    print(f"Non-Gemini averaged score, bootstrap over {len(clips)} clips\n")
    print(f"{'Model':26s} {'params':>8s} {'mean':>6s} {'95% CI':>18s}")
    for k, m in sorted(models.items(), key=lambda kv: -scores[kv[0]].mean()):
        v = scores[k]
        reps = rng.choice(v, size=(B, len(v)), replace=True).mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        p = m.get("params_B")
        print(f"{m['display'][:25]:26s} {('closed' if not p else f'{p:.1f}B'):>8s} "
              f"{v.mean():6.2f}  [{lo:.2f}, {hi:.2f}]")
        rows.append({"analysis": "nongemini_mean", "model": m["display"],
                     "params_B": p, "mean": round(float(v.mean()), 3),
                     "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                     "n_clips": len(clips)})

    # Paired differences: the correct comparison, since clips are shared.
    print("\nPaired difference vs Music Whisper (CI excluding 0 means significant)\n")
    mw = scores["music_whisper"]
    idx = rng.integers(0, len(clips), size=(B, len(clips)))
    for k, m in models.items():
        if k == "music_whisper":
            continue
        diff = mw - scores[k]
        reps = diff[idx].mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        sig = bool(lo > 0 or hi < 0)
        print(f"  MW - {m['display'][:24]:26s} {diff.mean():+.2f}  "
              f"[{lo:+.2f}, {hi:+.2f}]  {'significant' if sig else 'not significant'}")
        rows.append({"analysis": "paired_diff_vs_music_whisper", "model": m["display"],
                     "params_B": m.get("params_B"), "mean": round(float(diff.mean()), 3),
                     "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                     "significant": sig, "n_clips": len(clips)})

    # Original Gemini-judged Accuracy, reconstructed from published counts.
    print("\nGemini-judged per-snippet Accuracy, bootstrap from published counts\n")
    for name, counts in GEMINI_JUDGED_DIST.items():
        s = np.repeat(np.arange(1, 6), counts)
        reps = rng.choice(s, size=(B, len(s)), replace=True).mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        print(f"  {name:32s} n={len(s):3d}  {s.mean():.2f}  [{lo:.2f}, {hi:.2f}]")
        rows.append({"analysis": "gemini_judged_accuracy", "model": name,
                     "mean": round(float(s.mean()), 3), "ci_low": round(float(lo), 3),
                     "ci_high": round(float(hi), 3), "n_clips": len(s)})

    df = pd.DataFrame(rows)
    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / "caption_eval_bootstrap.csv", index=False)
    print("\nWrote output/caption_eval_bootstrap.csv")

    paired = df[df.analysis == "paired_diff_vs_music_whisper"]
    lines = [
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"\textbf{Comparison model} & \textbf{Params} & \textbf{$\Delta$ vs.\ ours}"
        r" & \textbf{95\% CI} & \textbf{Verdict} \\",
        r"\midrule",
    ]
    for _, r in paired.sort_values("mean").iterrows():
        p = "closed" if not r.params_B or np.isnan(r.params_B) else f"{r.params_B:.1f}B"
        verdict = "ours higher" if r.ci_low > 0 else ("ours lower" if r.ci_high < 0 else "tie")
        lines.append(f"{r.model} & {p} & ${r['mean']:+.2f}$ & "
                     f"$[{r.ci_low:+.2f}, {r.ci_high:+.2f}]$ & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "caption_eval_bootstrap_table.tex").write_text("\n".join(lines) + "\n")
    print("Wrote output/caption_eval_bootstrap_table.tex")

    gem = df[df.analysis == "gemini_judged_accuracy"].sort_values("mean", ascending=False)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{$n$} & \textbf{Mean Accuracy} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for _, r in gem.iterrows():
        lines.append(f"{r.model} & {int(r.n_clips)} & {r['mean']:.2f} & "
                     f"$[{r.ci_low:.2f}, {r.ci_high:.2f}]$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "caption_eval_gemini_ci_table.tex").write_text("\n".join(lines) + "\n")
    print("Wrote output/caption_eval_gemini_ci_table.tex")


if __name__ == "__main__":
    main()
