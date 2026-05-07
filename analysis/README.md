# Analysis — Perceptual Evaluation of AI-Generated Music

Statistical analysis pipeline for the NeurIPS 2026 paper. Covers human-annotation study results, LMM evaluation, automated aesthetic scoring, and audio-forensics feature experiments.

---

## Quick start

```bash
cd analysis
uv sync
uv run python main.py           # statistical analysis → output/
uv run python generate_report.py  # compile output/ into output/neurips_analysis_report.pdf
```

`main.py` reads human annotations from `input/v2/`, runs all statistical tests (inter-rater reliability, mixed-effects models, mood-tag profiles, model–human alignment), and writes tables, figures, and CSVs to `output/`. `generate_report.py` collects those artefacts into a single paginated PDF.

---

## LMM evaluation

`model_eval.py` queries multimodal language models against the same song set and writes per-model judgements to `input/models/`, which `main.py` then picks up for alignment analysis.

```bash
uv run python model_eval.py                  # all models in model_config.py
uv run python model_eval.py --model <id>     # single model
uv run python model_eval.py --limit 10       # smoke test
```

Model hyperparameters and provider routing live in `model_config.py`. Credentials go in `analysis/.env`:

```
HYPRLAB_API_KEY=...
HYPRLAB_API_ENDPOINT=https://.../v1/chat/completions
TRANSFORMERS_API_KEY=...
TRANSFORMERS_API_ENDPOINT=https://.../v1/chat/completions
```

---

## Sub-experiments

| Folder | What it contains |
|---|---|
| [`features/`](features/README.md) | Audio-forensics protocol: invariant probes (phase coherence, bicoherence, stereo coherence, …), augmentation ablations, OOD AUC benchmarks |
| [`aesthetics/`](aesthetics/README.md) | Audiobox and SongEval automated aesthetic scoring; setup instructions for third-party repos |

Each subfolder has its own `README.md` with detailed setup and execution instructions.

---

## Output artefacts

`main.py` populates `output/` with:

- LaTeX tables (`*.tex`) ready for inclusion in the paper
- Alignment CSVs (`*_alignment.csv`, `*_song_scores.csv`)
- Summary CSVs for mood tags, model–human comparisons, and feature ablations
- Figures (`output/figures/`)
- `analysis_report.txt` — plain-text narrative of key statistics

The final PDF report is written to `output/neurips_analysis_report.pdf`.

---

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). Requires Python ≥ 3.10.

```bash
uv sync   # installs everything from pyproject.toml
```

Key packages: `pandas`, `scipy`, `statsmodels`, `pingouin`, `scikit-learn`, `matplotlib`, `seaborn`, `krippendorff`, `fpdf2`.
