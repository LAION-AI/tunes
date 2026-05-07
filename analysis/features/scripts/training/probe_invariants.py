"""Step 3 — Q1: Probe invariants.

For each probe (phase, rolloff, bicoherence, chroma_ssm, denoiser, stereo,
mel_stats), fit a logistic regression on the training split and evaluate on
val + OOD splits.

Outputs:
  results/invariants/ranking.md    — probes ranked by OOD AUC
  results/summary.csv              — one row per probe-split

Usage:
  uv run scripts/training/probe_invariants.py --all
  uv run scripts/training/probe_invariants.py --probes phase,rolloff
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import (
    CACHE_DIR,
    RESULTS_DIR,
    build_song_manifest,
    compute_metrics,
    get_label,
    load_split,
    log_to_csv,
    save_scores,
)

FEATURE_DIR = CACHE_DIR / "features"

PROBES = ["phase", "rolloff", "bicoherence", "chroma_ssm", "denoiser", "stereo", "mel_stats"]


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _load_features(
    records: list[dict], probe: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns X (N, D), y (N,), uuids."""
    Xs, ys, uuids = [], [], []
    for entry in records:
        p = FEATURE_DIR / f"{entry['uuid']}.npz"
        if not p.exists():
            continue
        data = np.load(p)
        if probe not in data:
            continue
        feat = data[probe].astype(np.float32)
        if not np.isfinite(feat).all():
            feat = np.nan_to_num(feat)
        Xs.append(feat)
        ys.append(get_label(entry))
        uuids.append(entry["uuid"])
    if not Xs:
        raise RuntimeError(f"No cached features for probe='{probe}'")
    return np.stack(Xs), np.array(ys, dtype=np.int32), uuids


# ─── Single probe evaluation ──────────────────────────────────────────────────

def evaluate_probe(
    probe: str,
    train_records: list[dict],
    val_records: list[dict],
    ood_records: list[dict],
) -> dict[str, dict]:
    X_tr, y_tr, _ = _load_features(train_records, probe)
    X_val, y_val, _ = _load_features(val_records, probe)
    X_ood, y_ood, ood_uuids = _load_features(ood_records, probe)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)),
    ])
    clf.fit(X_tr, y_tr)

    results = {}
    for split_name, X, y, uuids in [
        ("train", X_tr, y_tr, None),
        ("val",   X_val, y_val, None),
        ("ood",   X_ood, y_ood, ood_uuids),
    ]:
        scores = clf.predict_proba(X)[:, 1]
        m = compute_metrics(scores, y)
        results[split_name] = m
        if uuids is not None:
            save_scores(np.array(scores), y, run_id=f"probe_{probe}", split=split_name)
        log_to_csv({
            "run_id": f"probe_{probe}",
            "model": f"probe_{probe}",
            "augment": "none",
            "seed": 0,
            "split": split_name,
            **m,
        })
    return results


# ─── Report generation ────────────────────────────────────────────────────────

def write_ranking(probe_results: dict[str, dict]) -> None:
    out_dir = RESULTS_DIR / "invariants"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranking.md"

    rows = []
    for probe, splits in probe_results.items():
        ood = splits.get("ood", {})
        val = splits.get("val", {})
        rows.append({
            "probe": probe,
            "ood_auc": ood.get("auc", 0.0),
            "ood_tpr_1pct": ood.get("tpr_at_1pct_fpr", 0.0),
            "val_auc": val.get("auc", 0.0),
        })
    rows.sort(key=lambda r: r["ood_auc"], reverse=True)

    theoretical_notes = {
        "phase": (
            "Neural vocoders (HiFi-GAN, BigVGAN) and latent-diffusion decoders are "
            "trained against magnitude/perceptual losses; phase is reconstructed "
            "implicitly and accumulates systematic group-delay errors."
        ),
        "rolloff": (
            "Many generators train on resampled audio (16/22/24 kHz) and vocoders "
            "have limited high-frequency modeling capacity, leaving energy deserts "
            "above 16–20 kHz."
        ),
        "bicoherence": (
            "Sample-wise/spectral losses don't explicitly preserve the nonlinear "
            "harmonic coupling that real instruments produce via physical resonance."
        ),
        "chroma_ssm": (
            "Autoregressive token models are constrained by context window; "
            "diffusion models trained on short clips lose long-range sectional "
            "coherence."
        ),
        "denoiser": (
            "Generative artifacts live precisely in the regions where the generative "
            "distribution diverges from the real-audio manifold; a simple spectral "
            "smoother exposes them as non-Gaussian residuals."
        ),
        "stereo": (
            "Many generators produce mono and upmix, or model channels with limited "
            "cross-channel consistency, breaking natural ILD/ITD cues."
        ),
        "mel_stats": (
            "Mel-band energy statistics capture overall spectral envelope differences; "
            "serves as a low-complexity baseline for probe comparison."
        ),
    }

    lines = [
        "# Q1 — Invariant Probe Ranking",
        "",
        "Probes ranked by OOD AUC (logistic regression head, linear by design).",
        "",
        "| Rank | Probe | OOD AUC | OOD TPR@1%FPR | Val AUC |",
        "|------|-------|---------|---------------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['probe']}` | {r['ood_auc']:.3f} | "
            f"{r['ood_tpr_1pct']:.3f} | {r['val_auc']:.3f} |"
        )

    lines += ["", "## Theoretical grounding", ""]
    for r in rows:
        note = theoretical_notes.get(r["probe"], "")
        lines += [
            f"### `{r['probe']}`",
            "",
            textwrap.fill(note, width=88),
            "",
        ]

    top2 = [r["probe"] for r in rows[:2]]
    lines += [
        "## Recommended top-2 for multiview model",
        "",
        f"`{top2[0]}` and `{top2[1]}`",
        "",
        "Feed these into `train_baseline.py --model multiview "
        f"--probes {','.join(top2)}`",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Ranking written → {out_path}")
    print(f"  Top 2 probes: {top2}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--probes", default=None,
                        help="Override probe list, e.g. phase,rolloff")
    args = parser.parse_args()

    probes = PROBES if args.all or args.probes is None else args.probes.split(",")

    manifest = build_song_manifest()
    train_recs = load_split("training", manifest)
    val_recs   = load_split("validation", manifest)
    ood_recs   = load_split("test-ood", manifest)

    all_results: dict[str, dict] = {}
    for probe in probes:
        print(f"\n── probe: {probe} ──")
        try:
            res = evaluate_probe(probe, train_recs, val_recs, ood_recs)
        except RuntimeError as exc:
            print(f"  [SKIP] {exc}")
            continue
        all_results[probe] = res
        ood_m = res.get("ood", {})
        val_m = res.get("val", {})
        print(
            f"  val_auc={val_m.get('auc', 0):.3f}  "
            f"ood_auc={ood_m.get('auc', 0):.3f}  "
            f"ood_tpr@1%={ood_m.get('tpr_at_1pct_fpr', 0):.3f}"
        )

    if all_results:
        write_ranking(all_results)


if __name__ == "__main__":
    main()
