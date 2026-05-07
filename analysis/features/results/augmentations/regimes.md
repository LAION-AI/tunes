# Q2 — Augmentation Ablation Results

LCNN backbone, fixed seed 42, 20 epochs. Regimes below baseline OOD AUC are **destroying the forensic signal**.

| Regime | Val AUC | OOD AUC | OOD TPR@1%FPR | Gap (val−ood) | Verdict |
|--------|---------|---------|---------------|---------------|---------|
| `none` | 0.842 | 0.975 | 0.678 | -0.133 | preserves signal |
| `specaug` | 0.837 | 0.980 | 0.831 | -0.143 | preserves signal |
| `mixup` | 0.827 | 0.970 | 0.542 | -0.143 | preserves signal |
| `codec` | 0.842 | 0.975 | 0.678 | -0.133 | preserves signal |
| `noise` | 0.845 | 0.972 | 0.678 | -0.127 | preserves signal |
| `pitch_shift` | 0.839 | 0.980 | 0.831 | -0.141 | preserves signal |
| `loudness` | 0.812 | 0.948 | 0.008 | -0.136 | **destroys signal** |
| `reverb` | 0.832 | 0.969 | 0.695 | -0.137 | **destroys signal** |
| `combined_safe` | 0.829 | 0.961 | 0.585 | -0.131 | **destroys signal** |
| `combined_aggressive` | 0.827 | 0.954 | 0.703 | -0.127 | **destroys signal** |

## Findings

**Signal-preserving regimes** (OOD AUC ≥ baseline): `none`, `specaug`, `mixup`, `codec`, `noise`, `pitch_shift`

**Signal-destroying regimes** (OOD AUC < baseline): `loudness`, `reverb`, `combined_safe`, `combined_aggressive`

> A regime that lifts Val AUC but drops OOD AUC is teaching the model a > shortcut (e.g. codec artifacts) rather than the true generative fingerprint.
