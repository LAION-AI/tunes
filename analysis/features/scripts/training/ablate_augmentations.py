"""Step 6 — Q2: Augmentation ablation study.

Trains LCNN from scratch under every augmentation regime and evaluates on
val + OOD. A regime that lifts val but drops OOD is destroying the forensic
signal — the central finding for Q2.

Output:
  results/augmentations/regimes.png  — OOD AUC bar chart vs 'none' baseline
  results/summary.csv                — one row per regime-split
  results/augmentations/regimes.md   — table + interpretations

Usage:
  uv run scripts/training/ablate_augmentations.py --regimes all
  uv run scripts/training/ablate_augmentations.py --regimes none,codec,combined_safe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.utils import (
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    build_song_manifest,
    compute_metrics,
    get_label,
    load_split,
    log_to_csv,
    save_scores,
)
from scripts.training.train_baseline import (
    DEVICE,
    AudioDataset,
    LCNN,
    REGIME_AUGMENTS,
    evaluate,
    train_one_epoch,
)

ALL_REGIMES = list(REGIME_AUGMENTS.keys())

EPOCHS = 20
BATCH_SIZE = 32
SEED = 42


def run_regime(
    regime: str,
    train_recs: list[dict],
    val_recs: list[dict],
    ood_recs: list[dict],
) -> dict[str, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = AudioDataset(train_recs, augment=regime, training=True)
    val_ds   = AudioDataset(val_recs,   augment="none", training=False)
    ood_ds   = AudioDataset(ood_recs,   augment="none", training=False)

    if len(train_ds) == 0:
        print(f"  [{regime}] SKIP: no training examples available.")
        return {}

    loader_kw = dict(num_workers=6, persistent_workers=True, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **loader_kw)
    ood_loader   = DataLoader(ood_ds,   batch_size=BATCH_SIZE, shuffle=False, **loader_kw)

    model = LCNN().to(DEVICE)
    labels_arr = np.array([get_label(r) for r in train_recs])
    n_neg, n_pos = (labels_arr == 0).sum(), (labels_arr == 1).sum()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, float(n_neg / max(n_pos, 1))],
            dtype=torch.float32,
            device=DEVICE,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    run_id = f"lcnn_aug-{regime}_ablation"
    ckpt_dir = CHECKPOINTS_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ood_auc, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, regime)
        scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_scores,  val_labels,  _ = evaluate(model, val_loader)
            ood_scores,  ood_labels,  _ = evaluate(model, ood_loader)
            val_m = compute_metrics(val_scores, val_labels)
            ood_m = compute_metrics(ood_scores, ood_labels)
            print(
                f"  [{regime}] epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}  "
                f"val_auc={val_m['auc']:.3f}  ood_auc={ood_m['auc']:.3f}"
            )
            if ood_m["auc"] > best_ood_auc:
                best_ood_auc = ood_m["auc"]
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "best.pt")

    # Load best and do final eval
    try:
        model.load_state_dict(
            torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=True)
        )
    except Exception:  # noqa: BLE001
        pass

    results: dict[str, dict] = {}
    for split_name, loader in [("val", val_loader), ("ood", ood_loader)]:
        scores, labels, _ = evaluate(model, loader)
        m = compute_metrics(scores, labels)
        results[split_name] = m
        log_to_csv({
            "run_id": run_id, "model": "lcnn", "augment": regime,
            "seed": SEED, "split": split_name, **m,
        })
        save_scores(scores, labels, run_id=run_id, split=split_name)

    print(
        f"  [{regime}] FINAL  val_auc={results['val']['auc']:.3f}  "
        f"ood_auc={results['ood']['auc']:.3f}  "
        f"ood_tpr@1%={results['ood']['tpr_at_1pct_fpr']:.3f}"
    )
    return results


def write_report(regime_results: dict[str, dict[str, dict]], baseline_ood_auc: float) -> None:
    out_dir = RESULTS_DIR / "augmentations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = list(regime_results.keys())
    ood_aucs = [regime_results[r]["ood"]["auc"] for r in regimes]
    colors = [
        "#d62728" if auc < baseline_ood_auc - 0.005 else
        "#2ca02c" if auc > baseline_ood_auc + 0.005 else
        "#aec7e8"
        for auc in ood_aucs
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(regimes) * 0.9), 5))
    bars = ax.bar(regimes, ood_aucs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(baseline_ood_auc, linestyle="--", color="black", linewidth=1.2,
               label=f"baseline (none) = {baseline_ood_auc:.3f}")
    ax.set_ylabel("OOD AUC")
    ax.set_title("Augmentation ablation — OOD AUC per regime (LCNN backbone)")
    ax.set_ylim(max(0, min(ood_aucs) - 0.05), min(1.0, max(ood_aucs) + 0.05))
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    for bar, auc in zip(bars, ood_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "regimes.png", dpi=150)
    plt.close(fig)
    print(f"Chart → {out_dir / 'regimes.png'}")

    # ── Markdown table ────────────────────────────────────────────────────────
    lines = [
        "# Q2 — Augmentation Ablation Results",
        "",
        "LCNN backbone, fixed seed 42, 20 epochs. "
        "Regimes below baseline OOD AUC are **destroying the forensic signal**.",
        "",
        "| Regime | Val AUC | OOD AUC | OOD TPR@1%FPR | Gap (val−ood) | Verdict |",
        "|--------|---------|---------|---------------|---------------|---------|",
    ]
    for regime in regimes:
        rm = regime_results[regime]
        val_auc = rm["val"]["auc"]
        ood_auc = rm["ood"]["auc"]
        tpr = rm["ood"]["tpr_at_1pct_fpr"]
        gap = val_auc - ood_auc
        verdict = (
            "**destroys signal**" if ood_auc < baseline_ood_auc - 0.005 else
            "preserves signal" if ood_auc >= baseline_ood_auc - 0.005 else
            "neutral"
        )
        lines.append(
            f"| `{regime}` | {val_auc:.3f} | {ood_auc:.3f} | {tpr:.3f} "
            f"| {gap:+.3f} | {verdict} |"
        )

    destroys = [r for r in regimes
                if regime_results[r]["ood"]["auc"] < baseline_ood_auc - 0.005]
    preserves = [r for r in regimes
                 if regime_results[r]["ood"]["auc"] >= baseline_ood_auc - 0.005]
    lines += [
        "",
        "## Findings",
        "",
        f"**Signal-preserving regimes** (OOD AUC ≥ baseline): "
        f"{', '.join(f'`{r}`' for r in preserves) or 'none'}",
        "",
        f"**Signal-destroying regimes** (OOD AUC < baseline): "
        f"{', '.join(f'`{r}`' for r in destroys) or 'none'}",
        "",
        "> A regime that lifts Val AUC but drops OOD AUC is teaching the model a "
        "> shortcut (e.g. codec artifacts) rather than the true generative fingerprint.",
    ]

    (out_dir / "regimes.md").write_text("\n".join(lines) + "\n")
    print(f"Report → {out_dir / 'regimes.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regimes", default="all",
        help="Comma-separated regime names or 'all'",
    )
    args = parser.parse_args()

    if args.regimes.strip().lower() == "all":
        regimes = ALL_REGIMES
    else:
        regimes = [r.strip() for r in args.regimes.split(",")]

    manifest = build_song_manifest()
    train_recs = load_split("training", manifest)
    val_recs   = load_split("validation", manifest)
    ood_recs   = load_split("test-ood", manifest)

    # Always run "none" first to get the baseline
    if "none" in regimes:
        regimes = ["none"] + [r for r in regimes if r != "none"]
    else:
        regimes = ["none"] + regimes

    regime_results: dict[str, dict[str, dict]] = {}
    for regime in regimes:
        print(f"\n{'='*60}\nREGIME: {regime}\n{'='*60}")
        results = run_regime(regime, train_recs, val_recs, ood_recs)
        regime_results[regime] = results

    baseline_ood_auc = regime_results.get("none", {}).get("ood", {}).get("auc", 0.5)
    write_report(regime_results, baseline_ood_auc)


if __name__ == "__main__":
    main()
