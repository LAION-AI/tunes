# Statistical comparison: proposed model vs. baselines

Split: **ood**. Bootstrap CIs: 1 000 resamples, 95% CI.
DeLong's paired AUC test. Wilcoxon signed-rank on per-clip scores.

## Proposed model

Run: `multiview_aug-combined_safe_seed`

| Metric | Value [95% CI] |
|--------|----------------|
| AUC | 0.979 [0.954, 0.996] |
| EER | 0.082 [0.023, 0.149] |
| TPR@1%FPR | 0.662 [0.468, 0.885] |
| TPR@0.1%FPR | 0.662 [0.468, 0.885] |

## Baseline comparisons

| Baseline | AUC [95% CI] | ΔAUC | DeLong z | DeLong p | Wilcoxon p | Significant |
|----------|-------------|------|---------|---------|-----------|-------------|
| `lcnn_aug-none_seed0` | 0.935 [0.872, 0.983] | +0.045 | 1.79 | 0.0729 | 0.0017 | — |
| `mert_head_aug-none_seed0` | 0.959 [0.925, 0.985] | +0.019 | 1.08 | 0.2808 | 0.0000 | — |
| `muq_head_aug-none_seed0` | 0.991 [0.978, 0.999] | -0.012 | -1.30 | 0.1948 | 0.0001 | ✗ |
| `moss_nano_head_aug-none_seed0` | 0.975 [0.950, 0.993] | +0.004 | 0.27 | 0.7834 | 0.4839 | — |
| `clap_head_aug-none_seed0` | 0.918 [0.862, 0.966] | +0.061 | 2.46 | 0.0141 | 0.0000 | ✓ p<0.05 |
| `convnext_aug-none_seed0` | 0.663 [0.583, 0.745] | +0.316 | 7.05 | 0.0000 | 0.0001 | ✓ p<0.05 |
| `vit_aug-none_seed0` | 0.689 [0.605, 0.775] | +0.291 | 6.22 | 0.0000 | 0.0000 | ✓ p<0.05 |
| `efficientvit_aug-none_seed0` | 0.786 [0.711, 0.855] | +0.193 | 5.18 | 0.0000 | 0.0000 | ✓ p<0.05 |
| `specttra_alpha_aug-none_seed0` | 0.792 [0.710, 0.864] | +0.188 | 5.12 | 0.0000 | 0.0013 | ✓ p<0.05 |
| `specttra_beta_aug-none_seed0` | 0.782 [0.700, 0.854] | +0.199 | 5.55 | 0.0000 | 0.0000 | ✓ p<0.05 |
| `specttra_gamma_aug-none_seed0` | 0.855 [0.784, 0.918] | +0.126 | 4.28 | 0.0000 | 0.0002 | ✓ p<0.05 |

## Interpretation

A result is considered statistically significant when:
- DeLong p < 0.05 (paired AUC comparison)
- Wilcoxon p < 0.05 (per-clip score differences)
- ΔAUC > 0 (proposed model is better)

If the proposed model does **not** beat all baselines with p < 0.05, the paper contribution lives in the Q1 + Q2 analysis, not the model. Reframe accordingly.
