# AI Music Detection — Research Protocol (M4 mini, 32GB)

Two questions drive this repo:

1. **What invariant property does real music have that current generators violate?**
2. **Which augmentations preserve the forensic signal vs. destroy it?**

Primary metrics: **OOD AUC** and **TPR @ 1% FPR**. In-distribution numbers are a sanity check, not the goal.

**NOTES**:
- use /Volumes/NVME/songrating-experiments as cache directory and for heavy data
- use source .venv/bin/activate and uv install / uv add, as well as uv run

---

## Hardware setup

- Device: `mps` for the model, CPU workers for data loading.
- DataLoader: `num_workers=6`, `persistent_workers=True`, `pin_memory=False` (MPS ignores it).
- Mixed precision: `torch.autocast("mps", dtype=torch.float16)` — ~40% memory saved.
- Batch sizes at 30 s, 16 kHz mono: CNN ≈ 32, SSM/Mamba ≈ 16, raw-waveform ≈ 8.
- Foundation models: load in fp16, freeze, run inference once and **cache embeddings to disk**. Never recompute per epoch.
- Cache mel / CQT / phase features to `./data/cache/` as `.npy` files. Recomputing per epoch will bottleneck.
- Watch `Activity Monitor → GPU History`; if memory pressure goes yellow, drop batch size before everything else.

## Expected layout

```
data/                          # symlink → /Volumes/NVME/songrating-data
scripts/training/
  cache_features.py
  cache_foundation_embeddings.py   # MERT, MuQ, MOSS-Nano, CLAP
  probe_invariants.py          # Q1
  train_baseline.py
  ablate_augmentations.py      # Q2
  stats_significance.py        # bootstrap CIs + Wilcoxon + DeLong
results/
  invariants/
  augmentations/
  baselines/
  summary.csv                  # one row per run, appended
```

---

## Q1 — Find an invariant

For each candidate property, build a **standalone** detector. If a feature alone yields non-trivial OOD AUC, it is an invariant the generators violate. Each probe is paired with the architectural reason it should work — this is the theoretical grounding reviewers will look for.

| Probe | What it measures | Why generators violate it |
|---|---|---|
| Phase coherence | Group-delay deviation across bands | Neural vocoders (HiFi-GAN, BigVGAN) and latent-diffusion decoders are trained against magnitude / perceptual losses; phase is reconstructed implicitly and accumulates systematic errors |
| High-band rolloff | Energy above 16 / 18 / 20 kHz | Many models train on resampled audio (16 / 22.05 / 24 kHz) and vocoders have limited high-frequency modeling capacity |
| Bicoherence | Higher-order spectral correlations between harmonics | Sample-wise / spectral losses don't explicitly preserve the nonlinear harmonic coupling that real instruments produce |
| Long-range self-similarity | Variance of chroma SSM over 30 s | Autoregressive token models (MusicGen, MusicLM) are constrained by context window length; diffusion models are typically trained on short clips and lose sectional coherence |
| Denoiser residual | Reconstruction error of an AE trained on **real only** | Generative artifacts live precisely in the regions where the generative distribution diverges from the real-audio manifold |
| Stereo coherence | L/R phase + amplitude relationship | Many models generate mono and upmix, or model channels with limited cross-channel consistency |

**Protocol (`probe_invariants.py`):**

1. Extract a low-dim feature vector per clip for each probe.
2. Fit a **logistic regression** head (sklearn). Linear by design — the feature is what's being tested, not the head.
3. Log: train AUC, val AUC, **OOD AUC**, **TPR @ 1% FPR**, **TPR @ 0.1% FPR**.
4. Rank probes by OOD AUC. Top 2 feed the multi-view model later.

Budget: each probe end-to-end in < 30 min on the M4. No deep encoders here.

---

## Foundation model baselines

The point of this section: **prove that handcrafted invariants beat black-box deep representations**, not just LCNN. If they don't, that's also a valid finding (and reshapes the paper accordingly).

Cache embeddings once, train a 2-layer MLP head on top. Frozen backbones only — no fine-tuning on M4.

| Model | Params | Domain | M4 viability |
|---|---|---|---|
| **MERT-v1-95M** | 95M | Music (purpose-built) | Comfortable, primary baseline |
| **MuQ** | ~300M | Music | Tight but doable, fp16 only |
| **MOSS-Audio-Tokenizer-Nano** | ~20M | General + music, 48 kHz stereo | Comfortable; use **encoder pre-quantization** features, not RVQ codes |
| **CLAP** (laion-clap) | ~150M | Audio-text contrastive | Comfortable, good for general audio |
| **EnCodec 24kHz** | ~15M | Neural codec | Comfortable, useful as an "older codec" reference point |

A note on MOSS: it's a tokenizer, not a representation model. The discrete RVQ codes are a lossy view; use the **continuous encoder output before quantization** as the embedding. The Nano variant is the right pick for M4 — the 1.6B full model won't batch usefully in 32GB.

---

## Q2 — Which augmentations preserve vs. destroy the signal

Fix the model (LCNN works as a yardstick), fix the seed, fix the epoch count (≈ 20). Vary only the augmentation regime.

| Regime | Contents |
|---|---|
| `none` | — |
| `specaug` | Time + frequency masking |
| `mixup` | α = 0.2, spectrogram domain |
| `codec` | MP3 64 / 128 / 192 kbps, OGG, AAC — applied **symmetrically** to real and fake |
| `noise` | RawBoost: linear + non-linear convolutive |
| `pitch_shift` | ± 2 semitones |
| `loudness` | Random gain ± 6 dB, LUFS normalize |
| `reverb` | Random IR convolution |
| `combined_safe` | specaug + codec + loudness |
| `combined_aggressive` | everything |

**Protocol (`ablate_augmentations.py`):**

1. Train from scratch per regime, log val + OOD AUC + TPR@1%FPR.
2. **A regime that lifts val but drops OOD is destroying signal.** That is the finding.
3. Always apply codec / loudness to both classes. Asymmetric application teaches the model to detect the codec, not the fake — the most common shortcut in published "AI detectors".
4. Output: bar chart of OOD AUC per regime, `none` as the dashed baseline. Anything below the baseline on OOD goes in the "destroys signal" bucket.

---

## Evaluation (every run, every model)

Single numbers lie. Always log to `results/summary.csv`:

- Val AUC, Val EER
- **OOD AUC**, OOD EER
- **TPR @ 1% FPR** ← real-world deployment metric
- **TPR @ 0.1% FPR** ← strict deployment metric (a flagged human artist is a lawsuit)
- Generalization gap = Val AUC − OOD AUC (lower = better)

### Statistical rigor (`stats_significance.py`)

"≥ 3 seeds" is not enough. For every reported number on the final model:

- **95% bootstrap confidence intervals** on AUC, EER, TPR@FPR. Resample the test set with replacement, 1000 iterations. Report as `0.847 [0.831, 0.862]`.
- **Paired Wilcoxon signed-rank test** between the proposed model and each baseline on per-clip scores. Report p-values.
- **DeLong's test** for AUC differences specifically — the standard for paired ROC comparisons.
- Run all final comparisons over **≥ 5 seeds**; report mean ± std alongside CIs.

Save raw per-clip scores for every final-model run so significance tests can be re-run without retraining.

---

## Run order

```bash
# 1. Cache handcrafted features (mel, CQT, phase, chroma)
uv run scripts/training/cache_features.py

# 2. Cache foundation model embeddings (one-time, ~few hours total)
uv run scripts/training/cache_foundation_embeddings.py --models mert,muq,moss_nano,clap

# 3. Q1 — all invariant probes
uv run scripts/training/probe_invariants.py --all

# 4. Foundation model baselines (frozen backbone + MLP head)
uv run scripts/training/train_baseline.py --model mert_head
uv run scripts/training/train_baseline.py --model muq_head
uv run scripts/training/train_baseline.py --model moss_nano_head
uv run scripts/training/train_baseline.py --model clap_head

# 5. LCNN reference + multi-view using top-2 probes from Q1
uv run scripts/training/train_baseline.py --model lcnn
uv run scripts/training/train_baseline.py --model multiview --probes phase,denoiser

# 6. Q2 — augmentation ablation (LCNN backbone, all regimes)
uv run scripts/training/ablate_augmentations.py --regimes all

# 7. Final: top invariants + signal-preserving augs + SWA + TTA, 5 seeds
uv run scripts/training/train_baseline.py \
    --model multiview \
    --probes phase,denoiser \
    --augment combined_safe \
    --swa --tta \
    --seeds 0,1,2,3,4

# 8. Statistical significance vs. all baselines
uv run scripts/training/stats_significance.py \
    --proposed results/multiview_final \
    --baselines lcnn,mert_head,muq_head,moss_nano_head,clap_head
```

---

## Definition of done

- `results/invariants/ranking.md` — probes ranked by OOD AUC + TPR@1%FPR, each with theoretical grounding noted. Answers Q1.
- `results/augmentations/regimes.png` — bar chart of OOD AUC per regime vs. `none`. Answers Q2.
- `results/baselines/comparison.md` — final model vs. LCNN + foundation model baselines, with bootstrap 95% CIs and Wilcoxon / DeLong p-values.
- The final model **must beat every baseline** on OOD AUC and TPR @ 1% FPR with **p < 0.05** (paired test) over 5 seeds. If it doesn't, the contribution lives in the analysis (Q1 + Q2 findings), not the model — still publishable, but reframe the paper around the negative-result story.