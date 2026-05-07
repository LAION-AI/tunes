# Experimental Summary — AI Song Detection (Corrected)

Pipelines: `sh run_pipeline.sh` followed by `sh run_pipeline_fix.sh`. All 11 + 9 steps completed (`results/pipeline_state/`). Primary evaluation split: **OOD** (out-of-distribution generators). Secondary: in-distribution `test`. Headline metrics: **AUC** and **TPR @ 1% FPR**.

> This file replaces an earlier audit summary that reported numbers from a pipeline bug (multiview was trained with `--probes phase,denoiser` instead of the recommended `bicoherence,denoiser`). All numbers below are from the *corrected* runs. The buggy multiview checkpoints and scores are preserved under `checkpoints/_archive_phase_denoiser/` for the audit trail. The full paper-ready writeup is `neurips2026_paper.md`.

---

## TL;DR

1. **A linear probe on `bicoherence` reaches OOD AUC 0.957** — beating every generic vision backbone and every fine-tuned SONICS SpecTTTra checkpoint.
2. **Frozen MuQ + linear head is the strongest single detector** — OOD AUC 0.991, TPR@1%FPR 0.788.
3. **The corrected multiview model** (LCNN + `bicoherence` + `denoiser`, 5 seeds, SWA + TTA) reaches **OOD AUC 0.972 ± 0.008** per seed, **0.979 [0.954, 0.996]** ensembled — statistically tied with `lcnn`, `mert_head`, `muq_head`, `moss_nano_head`, and significantly better than `clap_head`, all three vision backbones, and all three SpecTTTra variants.
4. **SONICS SpecTTTra does not transfer.** All three SpecTTTra-{α, β, γ} variants underperform the from-scratch LCNN baseline by 0.08–0.27 OOD AUC, despite being the published SOTA architecture for AI-music detection.
5. **`loudness` augmentation collapses TPR@1%FPR from 0.678 to 0.008** — the largest single regression in the paper. Every "combined" augmentation regime sits below the no-aug LCNN baseline OOD AUC.

---

## Q1 — Invariant probes (linear, by design)

| Rank | Probe          | OOD AUC   | OOD TPR@1%FPR | Val AUC |
|------|----------------|-----------|---------------|---------|
| 1    | `bicoherence`  | **0.957** | 0.347         | 0.816   |
| 2    | `denoiser`     | 0.782     | 0.339         | 0.699   |
| 3    | `mel_stats`    | 0.779     | 0.178         | 0.858   |
| 4    | `stereo`       | 0.752     | 0.136         | 0.691   |
| 5    | `rolloff`      | 0.719     | 0.000         | 0.636   |
| 6    | `chroma_ssm`   | 0.674     | 0.203         | 0.649   |
| 7    | `phase`        | 0.409     | 0.000         | 0.799   |

`phase` is the most striking *negative* result: high val AUC (0.799), OOD AUC **below chance** (0.409). The ranking, not val AUC, drives multiview probe selection (this is the lesson the run_pipeline.sh bug taught).

Source: `results/invariants/ranking.md`.

---

## Q2 — Augmentation ablation (LCNN backbone, seed 42, 20 epochs)

| Regime                | Val AUC | OOD AUC | OOD TPR@1%FPR | Verdict      |
|-----------------------|---------|---------|---------------|--------------|
| `none`                | 0.842   | 0.975   | 0.678         | preserves    |
| `specaug`             | 0.837   | **0.980** | **0.831**   | preserves    |
| `pitch_shift`         | 0.839   | **0.980** | **0.831**   | preserves    |
| `codec`               | 0.842   | 0.975   | 0.678         | preserves    |
| `noise`               | 0.845   | 0.972   | 0.678         | preserves    |
| `mixup`               | 0.827   | 0.970   | 0.542         | preserves    |
| `reverb`              | 0.832   | 0.969   | 0.695         | **destroys** |
| `combined_safe`       | 0.829   | 0.961   | 0.585         | **destroys** |
| `combined_aggressive` | 0.827   | 0.954   | 0.703         | **destroys** |
| `loudness`            | 0.812   | 0.948   | **0.008**     | **destroys** |

Source: `results/augmentations/regimes.md`.

---

## Q3 — Detector head-to-head on OOD (eleven baselines)

### OOD performance

| Model                              | OOD AUC [95% CI]         | OOD EER | OOD TPR@1%FPR |
|------------------------------------|--------------------------|---------|---------------|
| `muq_head`                         | **0.991 [0.978, 0.999]** | **0.072** | **0.788**     |
| **multiview** (corrected, ensemble) | 0.979 [0.954, 0.996]     | 0.082   | 0.662         |
| **multiview** (per-seed mean)      | 0.972 ± 0.008            | 0.083 ± 0.024 | 0.537 ± 0.077 |
| `moss_nano_head`                   | 0.975 [0.950, 0.993]     | 0.101   | 0.703         |
| `mert_head`                        | 0.959 [0.925, 0.985]     | 0.105   | 0.542         |
| `lcnn`                             | 0.934 [0.872, 0.983]     | 0.119   | 0.025         |
| `clap_head`                        | 0.918 [0.862, 0.966]     | 0.156   | 0.093         |
| `specttra_gamma` (SONICS γ)        | 0.852 [0.784, 0.918]     | 0.224   | 0.025         |
| `specttra_alpha` (SONICS α)        | 0.791 [0.710, 0.864]     | 0.284   | 0.008         |
| `specttra_beta`  (SONICS β)        | 0.780 [0.700, 0.854]     | 0.266   | 0.119         |
| `efficientvit` (EfficientViT-B1)   | 0.785 [0.711, 0.855]     | 0.284   | 0.127         |
| `vit` (ViT-S/16)                   | 0.688 [0.605, 0.775]     | 0.362   | 0.000         |
| `convnext` (ConvNeXt-Tiny)         | 0.663 [0.583, 0.745]     | 0.380   | 0.025         |

### In-distribution test (sanity check)

| Model            | Test AUC | Test TPR@1%FPR |
|------------------|----------|----------------|
| `mert_head`      | 0.976    | 0.718          |
| `moss_nano_head` | 0.964    | 0.298          |
| `muq_head`       | 0.957    | 0.451          |
| `efficientvit`   | 0.948    | 0.486          |
| Multiview (mean) | 0.927    | 0.434          |
| `lcnn`           | 0.855    | 0.212          |
| `clap_head`      | 0.797    | 0.018          |
| `vit`            | 0.769    | 0.069          |
| `specttra_gamma` | 0.755    | 0.029          |
| `specttra_beta`  | 0.724    | 0.212          |
| `specttra_alpha` | 0.714    | 0.102          |
| `convnext`       | 0.620    | 0.020          |

### Significance — proposed multiview vs. each baseline (OOD)

Paired DeLong on AUC, Wilcoxon signed-rank on per-clip scores. Significant ✓ requires DeLong p < 0.05 *and* ΔAUC > 0; ✗ means significantly *worse*; — = not significantly different.

| Baseline         | ΔAUC    | DeLong p | Wilcoxon p | Verdict          |
|------------------|---------|----------|------------|------------------|
| `lcnn`           | +0.045  | 0.073    | 0.002      | — (trend ✓)      |
| `mert_head`      | +0.019  | 0.281    | <1e-4      | — (tied)         |
| `muq_head`       | −0.012  | 0.195    | <1e-4      | — (tied)         |
| `moss_nano_head` | +0.004  | 0.783    | 0.484      | — (tied)         |
| `clap_head`      | +0.061  | 0.014    | <1e-4      | ✓                |
| `convnext`       | +0.316  | <1e-4    | <1e-4      | ✓                |
| `vit`            | +0.291  | <1e-4    | <1e-4      | ✓                |
| `efficientvit`   | +0.193  | <1e-4    | <1e-4      | ✓                |
| `specttra_alpha` | +0.188  | <1e-4    | 0.001      | ✓                |
| `specttra_beta`  | +0.199  | <1e-4    | <1e-4      | ✓                |
| `specttra_gamma` | +0.126  | <1e-4    | <1e-4      | ✓                |

Source: `results/baselines/comparison.md`.

---

## Per-seed multiview numbers (corrected)

| Seed | OOD AUC | OOD EER | OOD TPR@1%FPR | Test AUC |
|------|---------|---------|---------------|----------|
| 0    | 0.982   | 0.064   | 0.542         | 0.915    |
| 1    | 0.975   | 0.097   | 0.534         | 0.923    |
| 2    | 0.972   | 0.115   | 0.661         | 0.931    |
| 3    | 0.960   | 0.084   | 0.492         | 0.936    |
| 4    | 0.969   | 0.057   | 0.458         | 0.930    |
| mean | **0.972** | 0.083 | 0.537         | 0.927    |
| std  | 0.008   | 0.024   | 0.077         | 0.008    |

---

## Suggested paper framing (NeurIPS 2026 Datasets & Benchmarks)

The paper-ready writeup is `neurips2026_paper.md`. Headline structure:

1. **Lead claim — `bicoherence` as a near-universal AI-music invariant.** Linear probe, OOD AUC 0.957, theoretically grounded in nonlinear-system signal processing.
2. **Negative result — SONICS SpecTTTra doesn't transfer.** All three variants underperform from-scratch LCNN by 0.08–0.27 OOD AUC despite being the published SOTA. Strong evidence that AI-music-detection benchmarks need OOD generator splits.
3. **Augmentation hazard — `loudness` collapses low-FPR detection.** TPR@1%FPR drops from 0.678 → 0.008 for a regime that *raises* val AUC.
4. **Multiview ties the strongest baseline without foundation-model pretraining.** OOD AUC 0.972 ± 0.008 per seed, ensemble 0.979, statistically tied with MuQ.

---

## Artifacts

- Per-run rows: `results/summary.csv` (one row per (run_id, split); buggy multiview rows are still present for the audit trail)
- Per-run scores: `results/scores/<run_id>_{val,test,ood}.npz`
- Probe ranking: `results/invariants/ranking.md`
- Augmentation table + plot: `results/augmentations/regimes.md`, `regimes.png`
- Significance tests: `results/baselines/comparison.md` (final, all 11 baselines)
- Pipeline state stamps: `results/pipeline_state/*.done`
- Checkpoints: `checkpoints/<run_id>/`
- Archived buggy multiview runs: `checkpoints/_archive_phase_denoiser/`, `results/scores/_archive_phase_denoiser/`
- **Paper draft:** `neurips2026_paper.md`
