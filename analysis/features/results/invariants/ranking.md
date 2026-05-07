# Q1 — Invariant Probe Ranking

Probes ranked by OOD AUC (logistic regression head, linear by design).

| Rank | Probe | OOD AUC | OOD TPR@1%FPR | Val AUC |
|------|-------|---------|---------------|---------|
| 1 | `bicoherence` | 0.957 | 0.347 | 0.816 |
| 2 | `denoiser` | 0.782 | 0.339 | 0.699 |
| 3 | `mel_stats` | 0.779 | 0.178 | 0.858 |
| 4 | `stereo` | 0.752 | 0.136 | 0.691 |
| 5 | `rolloff` | 0.719 | 0.000 | 0.636 |
| 6 | `chroma_ssm` | 0.674 | 0.203 | 0.649 |
| 7 | `phase` | 0.409 | 0.000 | 0.799 |

## Theoretical grounding

### `bicoherence`

Sample-wise/spectral losses don't explicitly preserve the nonlinear harmonic coupling
that real instruments produce via physical resonance.

### `denoiser`

Generative artifacts live precisely in the regions where the generative distribution
diverges from the real-audio manifold; a simple spectral smoother exposes them as non-
Gaussian residuals.

### `mel_stats`

Mel-band energy statistics capture overall spectral envelope differences; serves as a
low-complexity baseline for probe comparison.

### `stereo`

Many generators produce mono and upmix, or model channels with limited cross-channel
consistency, breaking natural ILD/ITD cues.

### `rolloff`

Many generators train on resampled audio (16/22/24 kHz) and vocoders have limited high-
frequency modeling capacity, leaving energy deserts above 16–20 kHz.

### `chroma_ssm`

Autoregressive token models are constrained by context window; diffusion models trained
on short clips lose long-range sectional coherence.

### `phase`

Neural vocoders (HiFi-GAN, BigVGAN) and latent-diffusion decoders are trained against
magnitude/perceptual losses; phase is reconstructed implicitly and accumulates
systematic group-delay errors.

## Recommended top-2 for multiview model

`bicoherence` and `denoiser`

Feed these into `train_baseline.py --model multiview --probes bicoherence,denoiser`
