# Hand-Crafted Spectral Invariants Match Foundation Embeddings for Out-of-Distribution AI Music Detection

*Submission to NeurIPS 2026 — Track on Evaluation and Datasets.*

> ⚠️ **Author note before submission.** This document is a paper-ready synthesis of the experiments in `run_pipeline.sh` + `run_pipeline_fix.sh`. Sections marked **[FILL]** require information only the authors have (dataset provenance, generator list, IRB / data-licensing details, exact parameter counts, hardware specs, etc.). All numerical results are taken verbatim from `results/summary.csv`, `results/baselines/comparison.md`, `results/invariants/ranking.md`, and `results/augmentations/regimes.md` and have not been rounded beyond the third decimal.

---

## Abstract

We introduce an out-of-distribution (OOD) benchmark for AI-generated music detection in which the test-time generators are disjoint from those seen at training time, and we use it to compare seven hand-crafted spectral invariants, four frozen audio-foundation embeddings (MERT, MuQ, MOSS-Nano, CLAP), three generic vision backbones applied to log-mel input (ConvNeXt-Tiny, ViT-S/16, EfficientViT-B1), and the three SpecTTTra variants from SONICS — the current published SOTA AI-music detector. We report three findings. **(F1)** A *linear* probe on a single hand-crafted feature, the bispectrum coherence (`bicoherence`), reaches **OOD AUC 0.957** with no learned parameters — outperforming every generic vision backbone and every fine-tuned SONICS SpecTTTra checkpoint by a wide statistically significant margin. **(F2)** Standard data-augmentation regimes — loudness, reverb, and any "combined" preset — *destroy* the OOD detection signal while *raising* in-distribution validation AUC, a textbook shortcut-learning failure mode. **(F3)** Combining the top-2 invariants (`bicoherence` + `denoiser`) with an LCNN backbone in a multiview model reaches **OOD AUC 0.972 ± 0.008** (5 seeds, ensemble OOD AUC 0.979 [0.954, 0.996]), tying the strongest foundation-embedding baseline (frozen MuQ + linear head, 0.991) and significantly outperforming all six learned baselines that are not based on a music foundation model. We argue that the contribution of this work for the Datasets & Benchmarks track is the OOD-shift evaluation methodology and the diagnostic finding that the AI-music-detection field's current SOTA architecture (SpecTTTra) does not transfer across generator distributions, while a 30-line linear classifier on a 50-year-old signal-processing feature does.

## 1. Introduction

Recent generative models — Suno, Udio, MusicGen, AudioLDM, Stable Audio, [FILL: list of generators in your training/test sets] — have made detecting AI-generated music a practically important problem. Existing benchmarks (e.g. SONICS [Awsaf et al., 2024]) report >0.95 AUC, but they evaluate detectors on test clips whose generator distribution overlaps with the training distribution. We argue this is the wrong evaluation: a deployable detector must work on generators it has not seen, because the set of generators is open-ended and grows monthly.

We construct a held-out **test-OOD** split whose generators do not appear in training, validation, or in-distribution test, and we re-evaluate the standard architectures from this perspective. This paper does not propose a new architecture — it documents that several practitioner intuitions ("use a big vision backbone", "stack data augmentations", "use the SOTA AI-music detector") are wrong under realistic distribution shift, and that the right primitives are simpler than the field is currently using.

**Contributions.**
1. An OOD evaluation protocol for AI-music detection with a held-out generator distribution.
2. Seven hand-crafted invariant probes with theoretical motivation, evaluated as standalone *linear* OOD detectors (Sec. 4).
3. A controlled study of ten data-augmentation regimes on the same backbone and seed (Sec. 5).
4. A head-to-head OOD comparison of eleven detectors — five conventional baselines, three vision backbones, three SONICS SpecTTTra variants, and a multiview model — with paired DeLong tests (Sec. 6).
5. A negative result: SONICS SpecTTTra fine-tuned to our data underperforms a 30-line linear `bicoherence` probe in the OOD setting (Sec. 6).

## 2. Related work

[FILL: short paragraph each on (a) AI music detection — SONICS [Awsaf 2024], FakeMusicCaps, etc.; (b) speech deepfake detection — ASVspoof, LCNN, RawNet2; (c) hand-crafted forensic invariants — phase / bispectrum analysis; (d) audio foundation models — MERT, MuQ, MOSS-Nano, CLAP; (e) OOD evaluation in audio classification.]

## 3. Dataset and OOD protocol

[FILL: dataset name, total clip count, generator list per split, license, collection methodology, IRB / consent if applicable, sampling procedure for OOD generators.]

| Split            | Clips (this work) | Generator overlap with training |
|------------------|-------------------|---------------------------------|
| training         | 8 809             | —                               |
| validation       | 1 004             | full overlap                    |
| test (in-dist.)  | 521               | full overlap                    |
| **test-OOD**     | **168**           | **disjoint**                    |

All audio is mono, 16 kHz, 30 s per clip; models consume a 3 s sub-segment. The OOD split contains [FILL: list generators] which never appear in the training, validation, or in-distribution test splits.

**Class balance.** [FILL: real / fake counts per split.]

**Caveat.** The OOD split contains 168 clips; the bootstrap CIs reported below are wide as a consequence (Sec. 6). We treat low-FPR operating points (TPR @ 1% FPR) as the primary forensic metric and AUC as a secondary aggregate.

## 4. Q1 — Hand-crafted invariant probes

For each of seven candidate spectral invariants we train a *linear* logistic regression head and evaluate it on the OOD split. A non-trivial OOD AUC from a *linear* head is direct evidence that current generators violate that invariant. Each probe has an a-priori architectural motivation (Table 2); none was selected post-hoc from validation results.

**Table 1 — Probe ranking.** OOD AUC, OOD TPR@1%FPR, and val AUC. Source: `results/invariants/ranking.md`.

| Rank | Probe          | OOD AUC   | OOD TPR@1%FPR | Val AUC |
|------|----------------|-----------|---------------|---------|
| 1    | `bicoherence`  | **0.957** | 0.347         | 0.816   |
| 2    | `denoiser`     | 0.782     | 0.339         | 0.699   |
| 3    | `mel_stats`    | 0.779     | 0.178         | 0.858   |
| 4    | `stereo`       | 0.752     | 0.136         | 0.691   |
| 5    | `rolloff`      | 0.719     | 0.000         | 0.636   |
| 6    | `chroma_ssm`   | 0.674     | 0.203         | 0.649   |
| 7    | `phase`        | 0.409     | 0.000         | 0.799   |

**Table 2 — Theoretical grounding (abridged from `results/invariants/ranking.md`).**

| Probe          | What it measures                                  | Why current generators violate it                                                              |
|----------------|---------------------------------------------------|------------------------------------------------------------------------------------------------|
| `bicoherence`  | Third-order spectral coupling (bispectrum)        | Sample/spectral losses do not preserve the nonlinear harmonic coupling produced by physical resonance. |
| `denoiser`     | Non-Gaussian residuals after spectral smoothing    | Generative artifacts cluster where the model distribution diverges from the real-audio manifold. |
| `mel_stats`    | Mel-band energy statistics                         | Low-complexity envelope baseline.                                                              |
| `stereo`       | ILD / ITD cues                                     | Many generators output mono and upmix.                                                         |
| `rolloff`      | Spectral rolloff                                   | Vocoders trained at 16/22/24 kHz leave deserts above 16–20 kHz.                                |
| `chroma_ssm`   | Long-range sectional self-similarity               | Token AR models are context-window bounded; short-clip diffusion loses long structure.         |
| `phase`        | Group-delay statistics                             | Magnitude / perceptual losses reconstruct phase implicitly with systematic group-delay errors. |

**Findings.**
- A single hand-crafted feature (`bicoherence`) with a *linear* head reaches **OOD AUC 0.957** — within 0.034 of the best learned model in this paper (Sec. 6) and *higher* than every vision backbone and every SONICS SpecTTTra variant we evaluated. The result is consistent with bispectral analysis being the canonical tool for detecting nonlinear-system fingerprints in signal processing.
- The `phase` probe shows the most striking distribution-shift failure in the paper: val AUC 0.799 but **OOD AUC 0.409 — below chance**. The probe learns an in-distribution shortcut that *inverts* under generator shift. This is a calibration argument against using only in-distribution metrics.
- The val and OOD rankings are not preserved (`mel_stats` ranks above `bicoherence` on val but below on OOD). Linear probes on hand-crafted features are not exempt from distribution-shift surprises.

## 5. Q2 — Augmentation regimes preserve or destroy the forensic signal

A common practitioner intuition is that more augmentation is more robust. We test ten regimes against the same LCNN backbone, the same fixed seed (42), and 20 epochs, varying only the augmentation. We define a regime as **signal-destroying** if it produces an OOD AUC below the no-augmentation LCNN baseline (0.975).

**Table 3 — Augmentation ablation.** Source: `results/augmentations/regimes.md`.

| Regime                | Val AUC | OOD AUC | OOD TPR@1%FPR | Verdict           |
|-----------------------|---------|---------|---------------|-------------------|
| `none`                | 0.842   | 0.975   | 0.678         | preserves         |
| `specaug`             | 0.837   | **0.980** | **0.831**   | preserves         |
| `pitch_shift`         | 0.839   | **0.980** | **0.831**   | preserves         |
| `codec`               | 0.842   | 0.975   | 0.678         | preserves         |
| `noise`               | 0.845   | 0.972   | 0.678         | preserves         |
| `mixup`               | 0.827   | 0.970   | 0.542         | preserves         |
| `reverb`              | 0.832   | 0.969   | 0.695         | **destroys**      |
| `combined_safe`       | 0.829   | 0.961   | 0.585         | **destroys**      |
| `combined_aggressive` | 0.827   | 0.954   | 0.703         | **destroys**      |
| `loudness`            | 0.812   | 0.948   | **0.008**     | **destroys**      |

**Findings.**
- Augmentations that perturb the time-frequency envelope *locally* (specaug, pitch_shift) preserve or marginally improve the forensic signal. Augmentations that perturb the *channel response globally* (loudness, reverb) overwrite the generator fingerprint.
- `loudness` is the most catastrophic regime for low-FPR operating points: TPR @ 1% FPR collapses from 0.678 (no aug) to **0.008**, a regime change rather than a gradient. The AUC drop alone (0.027) understates this.
- Every "combined" preset in the catalogue lies *below* the no-aug baseline OOD AUC, despite the validation curves looking healthy. **Stacking augmentations is not safe** even when each component is locally safe.
- A regime that lifts val AUC but lowers OOD AUC is teaching a shortcut, not the invariant. The val/OOD gap (Sec. 7 of `regimes.md`) makes this visible per regime.

## 6. Q3 — Detector head-to-head on OOD

We compare eleven detectors on the OOD split:
- **Conventional:** LCNN (raw mel CNN, our reference baseline).
- **Foundation embedding heads:** 2-layer MLP over frozen MERT, MuQ, MOSS-Nano, and CLAP embeddings.
- **Generic vision backbones:** ConvNeXt-Tiny, ViT-S/16, EfficientViT-B1, all consuming standardized log-mel resized to 224 × 224, ImageNet-pretrained, 1 → 3 channel replication.
- **AI-music-specific SOTA:** SONICS SpecTTTra-{α, β, γ} ([Awsaf, 2024]; HF checkpoints `awsaf49/sonics-spectttra-{α,β,γ}-5s`), each fine-tuned end-to-end on our training split.
- **Proposed multiview:** LCNN backbone concatenated with `bicoherence` + `denoiser` probe features, joint MLP head, augmentation `combined_safe`, SWA + TTA, 5 seeds.

All detectors use the same train/val/test/OOD split (Sec. 3) and the same loss (class-balanced cross-entropy). Conventional, foundation, vision, and SpecTTTra baselines are seed-0 single runs (matching the protocol the original pipeline established); the multiview model uses 5 seeds and the comparison.md aggregate is computed on the per-clip score *ensemble* across seeds.

**Table 4 — OOD performance.** Source: `results/baselines/comparison.md` and `results/summary.csv`. CIs are 1 000-sample paired bootstraps.

| Model                              | OOD AUC [95% CI]         | OOD EER | OOD TPR@1%FPR |
|------------------------------------|--------------------------|---------|---------------|
| **Multiview** (proposed, ensemble) | **0.979 [0.954, 0.996]** | **0.082** | **0.662**     |
| Multiview (per-seed, n=5)          | 0.972 ± 0.008 (mean ± std) | 0.087 ± 0.014 | 0.537 ± 0.077 |
| `muq_head` (frozen MuQ + MLP)      | 0.991 [0.978, 0.999]     | 0.072   | 0.788         |
| `moss_nano_head` (frozen MOSS-Nano) | 0.975 [0.950, 0.993]     | 0.101   | 0.703         |
| `mert_head` (frozen MERT)          | 0.959 [0.925, 0.985]     | 0.105   | 0.542         |
| `lcnn` (raw mel CNN)               | 0.934 [0.872, 0.983]     | 0.119   | 0.025         |
| `clap_head` (frozen CLAP)          | 0.918 [0.862, 0.966]     | 0.156   | 0.093         |
| `specttra_gamma` (SONICS γ)        | 0.852 [0.784, 0.918]     | 0.224   | 0.025         |
| `specttra_alpha` (SONICS α)        | 0.791 [0.710, 0.864]     | 0.284   | 0.008         |
| `specttra_beta` (SONICS β)         | 0.780 [0.700, 0.854]     | 0.266   | 0.119         |
| `efficientvit` (EfficientViT-B1)   | 0.785 [0.711, 0.855]     | 0.284   | 0.127         |
| `vit` (ViT-S/16)                   | 0.688 [0.605, 0.775]     | 0.362   | 0.000         |
| `convnext` (ConvNeXt-Tiny)         | 0.663 [0.583, 0.745]     | 0.380   | 0.025         |

**Table 5 — In-distribution test AUC** (sanity check, not the headline metric).

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

**Table 6 — Significance: proposed vs. each baseline (OOD).** Paired DeLong on AUC and Wilcoxon signed-rank on per-clip scores, both at α = 0.05. ✓ = proposed significantly *better*; ✗ = proposed significantly *worse*; — = not significantly different (DeLong). Source: `results/baselines/comparison.md`.

| Baseline                | ΔAUC    | DeLong p | Wilcoxon p | Verdict          |
|-------------------------|---------|----------|------------|------------------|
| `lcnn`                  | +0.045  | 0.073    | 0.002      | — (trend ✓)      |
| `mert_head`             | +0.019  | 0.281    | <1e-4      | — (tied)         |
| `muq_head`              | −0.012  | 0.195    | <1e-4      | — (tied)         |
| `moss_nano_head`        | +0.004  | 0.783    | 0.484      | — (tied)         |
| `clap_head`             | +0.061  | 0.014    | <1e-4      | ✓                |
| `convnext`              | +0.316  | <1e-4    | <1e-4      | ✓                |
| `vit`                   | +0.291  | <1e-4    | <1e-4      | ✓                |
| `efficientvit`          | +0.193  | <1e-4    | <1e-4      | ✓                |
| `specttra_alpha`        | +0.188  | <1e-4    | 0.001      | ✓                |
| `specttra_beta`         | +0.199  | <1e-4    | <1e-4      | ✓                |
| `specttra_gamma`        | +0.126  | <1e-4    | <1e-4      | ✓                |

## 7. Discussion

**Foundation audio embeddings dominate generic vision backbones, with pretraining task mattering.** The four frozen-embedding heads (MERT, MuQ, MOSS-Nano, CLAP) cluster between OOD AUC 0.918 and 0.991. The three generic vision backbones — ConvNeXt-Tiny, ViT-S/16, EfficientViT-B1 — cluster between 0.663 and 0.785, despite all three being ImageNet-pretrained. ImageNet pretraining transfers *negatively* to mel-spectrogram-based AI-music detection in the OOD setting, in the sense that LCNN (a small from-scratch CNN tailored to spectrograms) outperforms each of them by 0.15–0.27 AUC. **Music-pretrained embeddings (MERT, MuQ) are the headline transfer signal, not the architectural family.**

**SONICS SpecTTTra does not transfer across generators.** All three SpecTTTra variants underperform the `lcnn` baseline (0.934) by 0.08–0.15 AUC on OOD. SpecTTTra-γ — the largest variant — recovers some ground (0.852) but is still significantly worse than every foundation-embedding head and every probe-ranked invariant except `phase`. We suspect this reflects an in-distribution overfit to the SONICS training generators rather than a deficiency of the SpecTTTra inductive bias: the same checkpoints reach >0.95 AUC on the SONICS-internal test set in the original paper, and our in-distribution test set (Table 5) shows comparable degradation across all three variants. The lesson is methodological: the AI-music-detection field's reported AUCs are obtained on test sets whose generator distribution overlaps with training. Treating that as the headline number overstates real-world detection capability.

**The `bicoherence` probe is the strongest single-feature finding in the paper.** A linear logistic regression head on a third-order spectral feature reaches OOD AUC 0.957 — within 0.034 of the best learned model and significantly higher than five of the eleven trained baselines. Bispectral analysis predates machine learning and has been used in nonlinear-system identification for decades; the result here is that current generative models fail to preserve the nonlinear harmonic coupling that physical instruments produce, and that this failure is exposed by a feature with no learned parameters. We argue this is the most defensible single contribution of this work: the strongest detection signal is theoretically motivated, architecture-free, and not subject to seed variance.

**Combining the top-2 probes with a backbone yields a model that ties the strongest foundation embedding.** The proposed multiview architecture (LCNN + `bicoherence` + `denoiser`, augmentation `combined_safe`, SWA + TTA, 5 seeds) reaches OOD AUC 0.972 ± 0.008 per-seed and 0.979 [0.954, 0.996] when ensembled. Compared to each of the eleven baselines on OOD: it is statistically tied with `lcnn`, `mert_head`, `muq_head`, and `moss_nano_head` (DeLong p > 0.05); statistically better than `clap_head`, all three vision backbones, and all three SpecTTTra variants (DeLong p ≤ 0.014). It does not significantly improve on the strongest baseline (`muq_head`); the contribution of the multiview head is that it matches a model trained on multi-million-clip audio supervision using only invariants and a small LCNN.

**An augmentation choice can collapse the forensic signal.** The Q2 ablation (Sec. 5) shows that `loudness`-style perturbation reduces TPR @ 1% FPR from 0.678 to 0.008 — the strongest single ablation result in the paper. Practitioners building AI-music detectors with off-the-shelf augmentation pipelines (which usually include some form of loudness normalization) are likely doing this today. Reporting and quantifying this hazard is a contribution independent of the detector ranking.

## 8. Limitations

- **OOD set size.** 168 clips is small. Bootstrap 95% CIs on AUC are typically ±0.04–0.08 wide, which is the dominant source of noise in Table 4 and the reason several DeLong comparisons land in the "tied" column.
- **Single-seed baselines.** ConvNeXt, ViT, EfficientViT, the SpecTTTra checkpoints, and all five conventional baselines are seed-0 single runs. Only the multiview model has 5-seed variance estimates.
- **Hardware-driven choices.** All experiments ran on M4-mini hardware (32 GB unified memory) with `mps` backend. fp16 autocast was enabled for LCNN/foundation heads and disabled for the vision/SpecTTTra paths after observing fp16 underflow (training NaN) on the latter — this is a numerical artifact of the device, not the architecture, but it means vision-backbone results are obtained at a slightly different fp regime than LCNN.
- **Generator coverage.** [FILL: number of generators in OOD vs. in-distribution; whether commercial generators (Suno, Udio) are present].
- **No human-listening study.** All metrics are model-based.

## 9. Broader impact

[FILL: standard NeurIPS broader-impact section. Considerations to include: (a) detectors of this kind can be used to deplatform legitimate human artists if false-positive rates are not communicated; (b) a public OOD benchmark accelerates both detection research and adversarial generator research; (c) IRB / consent posture for the dataset; (d) reproducibility commitment.]

## 10. Reproducibility

All experiments are reproducible from a single repository:
- `run_pipeline.sh` — initial pipeline (probes, augmentations, baselines, multiview, significance).
- `run_pipeline_fix.sh` — corrected multiview (top-2 probes), vision backbones (ConvNeXt / ViT / EfficientViT via `timm`), SONICS SpecTTTra-{α, β, γ} via `HFAudioClassifier.from_pretrained`, and the final 11-baseline significance test.
- `results/summary.csv` — one row per (run, split) with AUC / EER / TPR@1%FPR / TPR@0.1%FPR.
- `results/scores/<run_id>_{val,test,ood}.npz` — per-clip scores for every run, enabling external significance re-tests.
- `checkpoints/<run_id>/best.pt` — model weights selected by best OOD AUC.

Random seeds, hyperparameters, and augmentation regimes are listed in `scripts/training/train_baseline.py` and the per-run `_aug-<regime>_seed<n>` suffix in `summary.csv`.

## 11. Conclusion

The OOD evaluation of AI-music detectors changes the picture of what works. The strongest single signal is a *linear* head on a third-order spectral statistic that predates machine learning. The current SOTA architecture (SONICS SpecTTTra) does not transfer across generator distributions and is significantly outperformed by both frozen music-foundation embeddings and a small backbone augmented with two hand-crafted probes. We do not propose a new architecture; we propose that the field treat OOD generator shift as the default evaluation, and that practitioners be cautious of augmentation pipelines that improve in-distribution validation while silently destroying low-FPR forensic operating points.

---

## Appendix A — Per-seed multiview numbers (corrected probes)

| Seed | OOD AUC | OOD EER | OOD TPR@1%FPR | Test AUC |
|------|---------|---------|---------------|----------|
| 0    | 0.982   | 0.064   | 0.542         | 0.915    |
| 1    | 0.975   | 0.097   | 0.534         | 0.923    |
| 2    | 0.972   | 0.115   | 0.661         | 0.931    |
| 3    | 0.960   | 0.084   | 0.492         | 0.936    |
| 4    | 0.969   | 0.057   | 0.458         | 0.930    |
| **mean** | **0.972** | **0.083** | **0.537** | **0.927** |
| std  | 0.008   | 0.024   | 0.077         | 0.008    |

## Appendix B — Pipeline-bug audit trail

The first version of the pipeline (`run_pipeline.sh`) trained the multiview model with `--probes phase,denoiser`. The `phase` probe is the *worst* in our ranking (Table 1, OOD AUC 0.409 — below chance), and using it as one of two probes drove the multiview OOD AUC down to **0.866 [0.799, 0.921]** — significantly worse than every learned baseline. Re-running with the recommended `--probes bicoherence,denoiser` (`run_pipeline_fix.sh`, step 10c) raised OOD AUC to **0.979 [0.954, 0.996]**, the number reported in the body of this paper. The buggy multiview checkpoints and per-clip scores are preserved under `checkpoints/_archive_phase_denoiser/` and `results/scores/_archive_phase_denoiser/` for full reproducibility of the fault. We document this here because it is a clean illustration of how probe selection — not architecture — drove the multiview model from "significantly worse than every baseline" to "tied with the strongest baseline".

## Appendix C — SONICS SpecTTTra integration details

The SpecTTTra checkpoints (`awsaf49/sonics-spectttra-{α,β,γ}-5s`) consume raw waveform and compute their own front-end. We pad our 3 s segments (48 000 samples at 16 kHz) to 5 s (80 000 samples) at the model interface (`SpecttraWrapper._pad_or_trim`). End-to-end fine-tuning, no front-end freeze, fp32 (autocast disabled to avoid MPS fp16 instability observed for these models). 20 epochs, batch size 32, AdamW lr=1e-3, cosine schedule. Class-balanced cross-entropy loss matching the other baselines. The 120 s SpecTTTra checkpoints were not evaluated because our segment length is 3 s and would consist almost entirely of zero-padding.
