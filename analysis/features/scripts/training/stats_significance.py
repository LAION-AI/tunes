"""Step 8 — Statistical significance testing.

For the proposed model vs. each baseline, computes:
  1. 95% bootstrap confidence intervals on AUC, EER, TPR@1%FPR (1000 iterations)
  2. Paired Wilcoxon signed-rank test on per-clip scores
  3. DeLong's test for paired AUC comparison

Reads pre-saved .npz score files from results/scores/.
Requires ≥ 5 seeds for the proposed model.

Output:
  results/baselines/comparison.md

Usage:
  uv run scripts/training/stats_significance.py \\
      --proposed results/multiview_final \\
      --baselines lcnn,mert_head,muq_head,moss_nano_head,clap_head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import RESULTS_DIR, compute_metrics

SCORES_DIR = RESULTS_DIR / "scores"


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """Return {metric: (mean, lower, upper)} via non-parametric bootstrap."""
    rng = np.random.default_rng(0)
    n = len(scores)
    boot_auc, boot_eer, boot_tpr1, boot_tpr01 = [], [], [], []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            continue
        m = compute_metrics(s, l)
        boot_auc.append(m["auc"])
        boot_eer.append(m["eer"])
        boot_tpr1.append(m["tpr_at_1pct_fpr"])
        boot_tpr01.append(m["tpr_at_01pct_fpr"])

    alpha = (1 - ci) / 2

    def _ci(vals: list[float]) -> tuple[float, float, float]:
        arr = np.array(vals)
        return float(arr.mean()), float(np.percentile(arr, 100 * alpha)), \
               float(np.percentile(arr, 100 * (1 - alpha)))

    return {
        "auc":             _ci(boot_auc),
        "eer":             _ci(boot_eer),
        "tpr_at_1pct_fpr": _ci(boot_tpr1),
        "tpr_at_01pct_fpr": _ci(boot_tpr01),
    }


# ─── DeLong's test ────────────────────────────────────────────────────────────

def delong_auc_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Paired DeLong test. Returns (z_stat, p_value).

    Based on DeLong et al. (1988) — structural component method.
    """
    def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        m = len(y)
        mat = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if x[i] > y[j]:
                    mat[i, j] = 1.0
                elif x[i] == y[j]:
                    mat[i, j] = 0.5
        return mat

    pos = labels == 1
    neg = labels == 0
    x_a, y_a = scores_a[pos], scores_a[neg]
    x_b, y_b = scores_b[pos], scores_b[neg]

    m, n = len(x_a), len(y_a)
    if m == 0 or n == 0:
        return 0.0, 1.0

    V10_a = _kernel(x_a, y_a).mean(axis=1)  # (m,)
    V01_a = _kernel(x_a, y_a).mean(axis=0)  # (n,)
    V10_b = _kernel(x_b, y_b).mean(axis=1)
    V01_b = _kernel(x_b, y_b).mean(axis=0)

    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    # Covariance estimation
    S10 = np.cov(np.stack([V10_a, V10_b])) / m
    S01 = np.cov(np.stack([V01_a, V01_b])) / n

    var_a = S10[0, 0] + S01[0, 0]
    var_b = S10[1, 1] + S01[1, 1]
    cov   = S10[0, 1] + S01[0, 1]
    var_diff = var_a + var_b - 2 * cov

    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


# ─── Score loading ─────────────────────────────────────────────────────────────

def _load_scores(run_prefix: str, split: str = "ood") -> tuple[np.ndarray, np.ndarray] | None:
    """Load all seed files matching run_prefix, aggregate by mean score."""
    candidates = list(SCORES_DIR.glob(f"{run_prefix}*_{split}.npz"))
    if not candidates:
        return None
    all_scores, ref_labels = [], None
    for c in sorted(candidates):
        d = np.load(c)
        all_scores.append(d["scores"])
        if ref_labels is None:
            ref_labels = d["labels"]
    if ref_labels is None:
        return None
    # Ensemble by mean
    agg = np.stack(all_scores).mean(0)
    return agg, ref_labels


def _load_single(run_id: str, split: str = "ood") -> tuple[np.ndarray, np.ndarray] | None:
    p = SCORES_DIR / f"{run_id}_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["scores"], d["labels"]


# ─── Report ───────────────────────────────────────────────────────────────────

def write_comparison(
    proposed_id: str,
    baseline_ids: list[str],
    split: str = "ood",
) -> bool:
    out_dir = RESULTS_DIR / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    proposed = _load_scores(proposed_id, split) or _load_single(proposed_id, split)
    if proposed is None:
        print(f"[ERR] No score files found for proposed='{proposed_id}' split='{split}'")
        return False
    prop_scores, prop_labels = proposed
    prop_ci = bootstrap_ci(prop_scores, prop_labels)
    prop_m  = compute_metrics(prop_scores, prop_labels)

    def _fmt(ci_tuple: tuple[float, float, float]) -> str:
        m, lo, hi = ci_tuple
        return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = [
        "# Statistical comparison: proposed model vs. baselines",
        f"",
        f"Split: **{split}**. Bootstrap CIs: 1 000 resamples, 95% CI.",
        "DeLong's paired AUC test. Wilcoxon signed-rank on per-clip scores.",
        "",
        "## Proposed model",
        "",
        f"Run: `{proposed_id}`",
        "",
        f"| Metric | Value [95% CI] |",
        f"|--------|----------------|",
        f"| AUC | {_fmt(prop_ci['auc'])} |",
        f"| EER | {_fmt(prop_ci['eer'])} |",
        f"| TPR@1%FPR | {_fmt(prop_ci['tpr_at_1pct_fpr'])} |",
        f"| TPR@0.1%FPR | {_fmt(prop_ci['tpr_at_01pct_fpr'])} |",
        "",
        "## Baseline comparisons",
        "",
        "| Baseline | AUC [95% CI] | ΔAUC | DeLong z | DeLong p | Wilcoxon p | Significant |",
        "|----------|-------------|------|---------|---------|-----------|-------------|",
    ]

    for bl_id in baseline_ids:
        bl = _load_scores(bl_id, split) or _load_single(bl_id, split)
        if bl is None:
            lines.append(f"| `{bl_id}` | *no data* | — | — | — | — | — |")
            continue
        bl_scores, bl_labels = bl
        bl_ci = bootstrap_ci(bl_scores, bl_labels)
        bl_m  = compute_metrics(bl_scores, bl_labels)

        delta = prop_m["auc"] - bl_m["auc"]
        z, p_delong = delong_auc_test(prop_scores, bl_scores, prop_labels)

        # Wilcoxon on score differences (paired)
        min_len = min(len(prop_scores), len(bl_scores))
        diff = prop_scores[:min_len] - bl_scores[:min_len]
        try:
            _, p_wilcoxon = wilcoxon(diff)
        except Exception:  # noqa: BLE001
            p_wilcoxon = float("nan")

        sig = "✓ p<0.05" if (p_delong < 0.05 and delta > 0) else (
              "✗" if delta <= 0 else "—"
        )
        lines.append(
            f"| `{bl_id}` | {_fmt(bl_ci['auc'])} | {delta:+.3f} "
            f"| {z:.2f} | {p_delong:.4f} | {p_wilcoxon:.4f} | {sig} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "A result is considered statistically significant when:",
        "- DeLong p < 0.05 (paired AUC comparison)",
        "- Wilcoxon p < 0.05 (per-clip score differences)",
        "- ΔAUC > 0 (proposed model is better)",
        "",
        "If the proposed model does **not** beat all baselines with p < 0.05, "
        "the paper contribution lives in the Q1 + Q2 analysis, not the model. "
        "Reframe accordingly.",
    ]

    out_path = out_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Comparison written → {out_path}")
    return True


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposed", default="multiview_aug-combined_safe_seed",
        help="Prefix of proposed model run IDs (all seeds aggregated)",
    )
    parser.add_argument(
        "--baselines",
        default="lcnn_aug-none_seed0,mert_head_aug-none_seed0,"
                "muq_head_aug-none_seed0,moss_nano_head_aug-none_seed0,"
                "clap_head_aug-none_seed0",
        help="Comma-separated run IDs",
    )
    parser.add_argument("--split", default="ood")
    args = parser.parse_args()

    baseline_ids = [b.strip() for b in args.baselines.split(",")]
    if not write_comparison(args.proposed, baseline_ids, args.split):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
