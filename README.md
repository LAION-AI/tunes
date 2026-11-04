# LAION-Tunes Suite

> Code and analysis for the NeurIPS 2026 Evaluations & Datasets Track paper *LAION-Tunes: An Open 1.4M-Track Dataset and Perceptual Benchmark for AI-Generated Music* (Kaczmarczyk, Ahmed, Friedrich, Erickson, Sharoni, Herremans, Schuhmann).

[![Code: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Data: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Tracks: 1.4M](https://img.shields.io/badge/Tracks-1%2C429%2C734-brightgreen)]()
[![Benchmark Songs: 10,521](https://img.shields.io/badge/Benchmark%20Songs-10%2C521-orange)]()
[![Human Annotations: 591](https://img.shields.io/badge/Human%20Annotations-591-red)]()
[![Platforms: 7](https://img.shields.io/badge/AI%20Platforms-7-purple)]()

---

## Overview

**LAION-Tunes Suite** is a large-scale, richly annotated ecosystem for AI-generated music research, released as interoperable artifacts:

| Artifact | Description | Size |
|---|---|---|
| [LAION-Tunes Corpus](https://huggingface.co/datasets/laion/laion-tunes) | Annotated metadata for 1.4M AI-generated tracks from Suno, Udio & Mureka | 1,429,734 tracks |
| [LAION-Tunes-Benchmark](https://huggingface.co/datasets/laion/laion-tunes-benchmark) | Perceptual benchmark for AI music detection & quality assessment | 10,521 songs / 591 human annotations |
| [LAION-Tunes-RPG](https://huggingface.co/datasets/laion/laion-tunes-rpg-music) | Situation-aware instrumental subset annotated across 18 RPG genres | 2,580 tracks |
| [Music-Whisper](https://huggingface.co/laion/music-whisper) | Music captioning model (Whisper Small fine-tune) | 922 MB |
| [Music-Aesthetics](https://huggingface.co/laion/music-aesthetics) | SongEval-calibrated five-dimension quality scorer | 5 expert heads |
| [Music-Popularity](https://huggingface.co/laion/music-popularity) | Play and upvote count predictor | 91 MB |
| [Tunes-Search](https://projects.laion.ai/tunes/search-tunes/) | Public web demo over the corpus | |

Together, these resources enable research on text-to-music generation quality, authenticity detection, music retrieval, and human perception of AI-generated audio — at a scale previously unavailable to the open research community.

---

## Motivation

The rapid proliferation of AI music generators (Suno, Udio, Mureka, Lyria, …) has outpaced the availability of structured, richly annotated datasets for studying them. Existing benchmarks are small, cover only one or two generators, lack human perceptual data, or rely on proprietary audio that cannot be redistributed.

**LAION-Tunes** closes this gap by providing:

1. **Scale** — 1.4M tracks spanning three major platforms, ~68,000 hours of audio.
2. **Rich annotation** — captions, ASR transcription embeddings, five-dimensional aesthetics scores, multi-tier NSFW labels, and 768-dim text and audio embeddings.
3. **Human grounding** — a dedicated benchmark with 591 perceptual annotation trials from 61 participants across 7 AI platforms + real music.
4. **Ready-to-use retrieval infrastructure** — pre-built FAISS and BM25 indices served by a FastAPI search server with a full-featured web UI.
5. **Downstream subsets** — a curated RPG music collection annotated by a frontier LLM for situation-based search and game audio use cases.

---

## Key Numbers at a Glance

```
LAION-Tunes Corpus
├── 1,429,734 tracks  (Suno 1,037,381 · Mureka 383,549 · Udio 8,804)
├── ~68,471 hours of AI-generated audio
├── 1,356,009 Music-Whisper captions  (94.8%)
├── 1,041,488 Parakeet transcription embeddings  (72.8%; text not released)
├── 6 FAISS vector indices + 2 BM25 text indices
└── 36-column SQLite database, fully reproducible

LAION-Tunes-Benchmark
├── 10,521 songs  (9,681 AI · 840 human Apple Music previews)
├── 7 AI platforms  (Suno · Udio · Mureka · Sonauto · Lyria 3 · Riffusion · SilverknightAI)
├── 4 splits  (train / validation / test / test_ood)
├── 591 annotation trials · 61 participants
└── 6 perceptual dimensions  (authenticity + 5 quality ratings, 1–10 scale)

LAION-Tunes-RPG
├── 2,580 instrumental tracks  (Suno 1,489 · Udio 1,091)
├── 18 RPG genres  (High Fantasy → Alternate History)
├── 39,605 situation vectors  (Gemini 3 Flash Preview annotations)
└── 7 search modes  (vector · BM25 · audio upload)
```

---

## Corpus: LAION-Tunes

> [https://huggingface.co/datasets/laion/laion-tunes](https://huggingface.co/datasets/laion/laion-tunes)

### Source Data

Metadata was collected from three publicly accessible AI music generation platforms:

- **Suno** (1,037,381 tracks): Profiles discovered via the community-crawled Suno user dataset. Metadata (title, tags, play count, duration) collected from public profiles.
- **Mureka** (383,549 tracks): Metadata from publicly available tracks on the Mureka platform. Generation timestamps recoverable from CDN URLs.
- **Udio** (8,804 tracks): Audio URLs and metadata sourced from the public `blanchon/udio_dataset`.

> **Audio files are not distributed.** The dataset contains metadata, annotations, and embeddings. All `audio_url` fields point to the originating platform CDNs.

### Annotation Pipeline

Each track was processed through a seven-stage pipeline:

| Stage | Model | Output |
|---|---|---|
| 1. Music Captioning | `laion/music-whisper` (Whisper Small) | Free-text music description |
| 2. ASR Transcription | `nvidia/parakeet-tdt-0.6b-v3` | Transcript, embedded with EmbeddingGemma; the text is not released |
| 3. Text Embeddings | `google/embeddinggemma-300m` | 768-dim L2-normalised vectors (tags, caption, transcription, lyric, mood) |
| 4. Audio Embeddings | `laion/music-whisper` encoder | 768-dim mean-pooled Whisper encoder hidden states |
| 5. Aesthetics Scoring | SongEval MLP | 5-head MLP: coherence, musicality, memorability, clarity, naturalness (1–5) |
| 6. NSFW Classification | Cosine similarity vs. reference prompts | Three-tier labels × three dimensions (gore, sexual, hate speech) |
| 7. Language Detection | `langdetect` on ASR | ISO 639-1 language code |

### Dataset Statistics

| Metric | Value |
|---|---|
| Total tracks | 1,429,734 |
| Total audio duration | ~68,471 hours |
| Has caption | 1,356,009 (94.8%) |
| Has ASR transcription embedding | 1,041,488 (72.8%) |
| Instrumental | 388,301 (27.2%) |
| NSFW flagged (≥likely) | 23,591 (1.65%) |
| Average aesthetics score | 3.29 / 5.0 |
| FAISS vector indices | 6 (whisper, caption, transcription, tag, lyric, mood) |
| BM25 text indices | 2 (caption, tags) |

### Aesthetics Scores by Platform

| Platform | Coherence | Musicality | Memorability | Clarity | Naturalness | **Average** |
|---|---|---|---|---|---|---|
| Suno | 3.45 | 3.26 | 3.35 | 3.17 | 3.15 | **3.28** |
| Mureka | 3.54 | 3.34 | 3.42 | 3.22 | 3.18 | **3.34** |
| Udio | 3.29 | 3.04 | 3.14 | 2.98 | 2.94 | **3.08** |

### Search Infrastructure

The corpus ships with a production-ready FastAPI search server and a dark-mode single-page web UI offering:

- **Simple mode**: query bar with language filters and negative prompting.
- **Advanced mode**: vector similarity (caption / tag / lyric / mood / transcription), BM25 text, combined retrieval, two-stage refinement, and audio-upload similarity search.

```bash
# Quick start (requires HF_TOKEN for google/embeddinggemma-300m)
pip install fastapi uvicorn faiss-cpu numpy pandas sentence-transformers torch \
            scipy tqdm python-multipart transformers
HF_TOKEN=your_token python server.py --port 7860
# → http://localhost:7860
```

---

## Model: Music-Whisper

> [https://huggingface.co/laion/music-whisper](https://huggingface.co/laion/music-whisper)

A fine-tuned version of **OpenAI Whisper Small** trained for music captioning. The model generates detailed, paragraph-length descriptions of audio tracks, covering:

- Instrumentation and timbre
- Tempo and rhythmic structure
- Vocal characteristics (range, delivery, presence)
- Genre and mood
- Production quality
- Suggested use-case contexts

Music-Whisper was used to caption **1,356,009 tracks** in the LAION-Tunes corpus (94.8% coverage), and its mean-pooled encoder hidden states form the `faiss_whisper` index enabling audio-upload similarity search.

**License**: Apache 2.0

---

## Subset: LAION-Tunes-RPG

> [https://huggingface.co/datasets/laion/laion-tunes-rpg-music](https://huggingface.co/datasets/laion/laion-tunes-rpg-music)

A curated subset of **2,580 instrumental tracks** from the LAION-Tunes corpus (Suno + Udio), annotated by **Gemini 3 Flash Preview** with structured RPG gameplay situations across 18 genre categories.

### The 18 RPG Genres

High Fantasy · Low/Gritty Fantasy · Dark Fantasy · Mythic/Ancient World · Medieval Historical · Renaissance/Pirate Age · Wild West · Gothic Horror · Cosmic Horror · Modern Supernatural · Modern Realistic · Superhero · Post-Apocalyptic · Cyberpunk · Hard Sci-Fi · Space Opera · Science Fantasy · Alternate History

### Annotation Schema

Each track was annotated at full audio length with Gemini 3 Flash Preview, producing:

```json
{
  "has_singing": "no",
  "evoked_emotions": ["tension", "mystery", "foreboding"],
  "genre_situations": {
    "Dark Fantasy": [
      "Traversing a cursed land where shadows move independently",
      "Entering a necromancer's tower during a blood moon"
    ]
  }
}
```

### Statistics

| Metric | Value |
|---|---|
| Total tracks | 2,580 |
| Instrumental (no vocals) | 2,075 (80.4%) |
| Total RPG genres | 18 |
| Total situation vectors | 39,605 |
| Average duration | 183.4 s (~3 min) |
| Annotation cost | ~$41.59 (Gemini 3 Flash Preview) |

```bash
pip install fastapi uvicorn faiss-cpu numpy scipy sentence-transformers onnxruntime pandas
python rpg_server.py --port 7862
# → http://localhost:7862
```

---

## Benchmark: LAION-Tunes-Benchmark

> [https://huggingface.co/datasets/laion/laion-tunes-benchmark](https://huggingface.co/datasets/laion/laion-tunes-benchmark)

A **10,521-song perceptual evaluation dataset** for AI music detection and quality assessment, anchored in the LAION-Tunes corpus.

### Dataset Construction

The benchmark pairs AI-generated tracks from **seven commercial platforms** with **840 human Apple Music previews** (30-second clips of commercially released songs, sourced via the Genius metadata index):

| Split | Human | Suno | Udio | Mureka | Sonauto | OOD |
|---|---|---|---|---|---|---|
| train | 685 | 2,131 | 2,145 | 2,123 | 1,742 | — |
| validation | 74 | 243 | 242 | 248 | 198 | — |
| test | 31 | 126 | 113 | 129 | 123 | — |
| test_ood | 50 | — | — | — | — | Lyria 3 (51) · Riffusion (51) · SilverknightAI (16) |
| **Total** | **840** | **2,500** | **2,500** | **2,500** | **2,063** | **118** |

Suno, Udio, and Mureka tracks are randomly sampled from the LAION-Tunes large-scale corpus. Sonauto, human, and OOD sources were manually collected and are exclusive to this benchmark.

### Human Annotation Study

Annotations were collected via a purpose-built web platform across two phases (591 trials, 61 participants):

- **Phase 1** (403 trials, 2026-04-07): AI-only pool, full-length audio. Tests blind AI-detection from learned acoustic heuristics.
- **Phase 2** (188 trials, 2026-04-20): Balanced 50/50 human/AI design, 30-second clips (uniformly cropped for AI, fixed preview for human).

Each trial collected:
1. **Authenticity assessment**: `real` / `ai-generated` / `uncertain`
2. **Familiarity**: `never` / `heard_before` / `know_well`
3. **Five perceptual dimensions** on continuous 1–10 sliders: aesthetic quality, production quality, emotional engagement, musical creativity, playlist likelihood
4. Free-text AI-aspect tags, mood labels, and aesthetic comments

#### Authenticity Detection Accuracy

| Source | Trials | Correct | Accuracy |
|---|---|---|---|
| Human | 84 | 54 | **64.3%** |
| Suno | 133 | 74 | 55.6% |
| Mureka | 133 | 76 | 57.1% |
| Sonauto | 125 | 69 | 55.2% |
| **Udio** | 116 | 48 | **41.4%** |

> **Key finding**: songs judged real receive nearly two more aesthetic-quality points than songs judged AI-generated, within both actually-AI and actually-human songs, and the same pattern appears in frontier LLM judges. This *quality–authenticity halo* is an association, not a causal claim; quality ratings collected next to a provenance judgment are not independent.

#### Perceptual Quality Ratings (1–10 scale, mean ± σ)

| Source | Aesthetic | Production | Emotional | Creativity | Playlist |
|---|---|---|---|---|---|
| Human | 5.68 ± 1.94 | 6.06 ± 1.69 | 5.02 ± 1.91 | 5.42 ± 2.02 | 4.31 ± 2.02 |
| Suno | 5.18 ± 2.31 | 5.34 ± 2.34 | 4.79 ± 2.21 | 4.70 ± 2.18 | 3.99 ± 2.42 |
| Udio | 5.15 ± 1.98 | 5.21 ± 2.10 | 4.78 ± 2.06 | 4.83 ± 2.10 | 3.85 ± 2.15 |
| Mureka | 5.02 ± 2.22 | 5.19 ± 2.12 | 4.32 ± 2.13 | 4.36 ± 2.09 | 3.45 ± 2.20 |
| Sonauto | 4.94 ± 2.38 | 5.18 ± 2.54 | 4.56 ± 2.30 | 4.81 ± 2.25 | 3.69 ± 2.37 |

### Quick Start

```python
import datasets

# Gated: accept the access conditions on the dataset page and run `hf auth login` first.
ds = datasets.load_dataset("laion/laion-tunes-benchmark")

# Test split — every row has human annotations
test = ds["test"]
sample = test[0]
print(sample["title"], sample["source"])
print(sample["annotations"]["authenticity_assessment"])  # list of human labels
print(sample["annotations"]["aesthetic_quality"])        # list of 1–10 ratings

# OOD split — Lyria 3, Riffusion, SilverknightAI + rare human songs
test_ood = ds["test_ood"]
```

---

## Ethical Considerations

### Audio and Participant Rights

- **Human reference tracks** are 30-second Apple Music open previews distributed under the terms of Apple Music's public preview access. No full tracks are included.
- **AI-generated tracks** are served via public CDN links from the originating platforms; availability is subject to each platform's terms of service.
- All **annotation participants** provided informed consent for anonymous open data sharing. Each participant is identified only by an opaque UUID; no name, email, or other directly identifying information is released.

### Intended Use

This dataset is intended for academic research on AI music detection, perceptual quality modelling, retrieval, and human–AI comparative evaluation. It is **not** intended for building commercial AI-music detection products without further validation, deanonymising participants or artists, or inferring personally sensitive attributes from audio features.

### Bias and Fairness

The human reference pool is drawn from commercially indexed music (Apple Music via Genius), over-representing mainstream Western genres. The AI generator pool reflects platforms most prevalent on the open web as of early 2026. The participant pool (n = 61, mean age 29.4) skews toward music-engaged, AI-familiar individuals and may not generalise to broader listener populations.

### NSFW Safety

The corpus includes three-tier NSFW labels (gore/violence, sexual content, hate speech) with raw cosine similarity scores, enabling researchers to apply custom thresholds. The classifier uses English-only reference prompts — non-English NSFW content is likely under-flagged (~1.65% overall flagged rate). Raw scores are released to support calibration studies.

### Environmental Impact

No new audio was generated for this benchmark. All AI tracks were collected from publicly accessible platform outputs. The annotation pipeline (Music-Whisper captioning, Parakeet ASR, Gemini RPG annotations) was run once and is not repeated at inference time.

---

## Limitations

- Audio files are not distributed; CDN URLs may expire.
- Annotation coverage for the benchmark is partial (591 trials / 572 of 10,521 songs, ~5.4%).
- The NSFW classifier uses English-only reference prompts; non-English content is likely under-flagged.
- Platform engagement metrics (play count, upvote count) are unavailable for Mureka and partially imputed for extended Suno tracks.
- Benchmark participants (n = 61, mean age 29.4) are a convenience sample not representative of the general population.
- The benchmark reflects model capabilities as of April 2026.

---

## Repository Structure

The datasets and models live on Hugging Face (links above). This repository holds the code:

```
tunes/
├── analysis/            # Paper analyses: main.py, model_eval.py and the camera-ready scripts
│   ├── input/           # Annotation exports, benchmark subsets and model outputs (user-level fields removed)
│   ├── output/          # Tables and figures used in the paper
│   ├── features/        # Audio-feature AI-music detectors
│   └── aesthetics/      # Audio-aesthetics comparisons
└── docs/                # Project page and the Tunes-Search front end
```

---

## Citation

```bibtex
@inproceedings{kaczmarczyk2026laiontunes,
  title     = {{LAION-Tunes}: An Open 1.4{M}-Track Dataset and Perceptual Benchmark for {AI}-Generated Music},
  author    = {Kaczmarczyk, Robert and Ahmed, Tawsif and Friedrich, Felix and Erickson, Aidan C. and Sharoni, Orian and Herremans, Dorien and Schuhmann, Christoph},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

---

## License

| Artifact | License |
|---|---|
| Code in this repository | Apache 2.0 ([LICENSE](LICENSE)) |
| LAION-Tunes Corpus | CC BY-NC 4.0, gated research access |
| LAION-Tunes-Benchmark | CC BY-NC 4.0 song metadata; CC BY 4.0 human annotations |
| Music-Whisper, Music-Popularity | Apache 2.0 |
| Music-Aesthetics | CC BY-NC-SA 4.0, inherited from SongEval |
| LAION-Tunes-RPG | CC BY-NC 4.0 |

Data under `analysis/input` follows the benchmark's terms.
