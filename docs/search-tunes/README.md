---
language:
- en
license: apache-2.0
size_categories:
- 1M<n<10M
task_categories:
- audio-classification
- text-to-audio
- feature-extraction
tags:
- music
- ai-generated-music
- audio
- embeddings
- search
- faiss
- bm25
- nsfw-detection
- transcription
- captioning
pretty_name: REDACTED-Tunes
dataset_info:
  features:
  - name: row_id
    dtype: int64
  - name: audio_url
    dtype: string
  - name: filename
    dtype: string
  - name: tar_url
    dtype: string
  - name: subset
    dtype: string
  - name: title
    dtype: string
  - name: tags_text
    dtype: string
  - name: mood_text
    dtype: string
  - name: has_lyrics
    dtype: bool
  - name: genre_tags
    dtype: string
  - name: scene_tags
    dtype: string
  - name: emotion_tags
    dtype: string
  - name: score_coherence
    dtype: float64
  - name: score_musicality
    dtype: float64
  - name: score_memorability
    dtype: float64
  - name: score_clarity
    dtype: float64
  - name: score_naturalness
    dtype: float64
  - name: score_average
    dtype: float64
  - name: play_count
    dtype: int64
  - name: upvote_count
    dtype: int64
  - name: duration_seconds
    dtype: float64
  - name: music_whisper_caption
    dtype: string
  - name: parakeet_transcription
    dtype: string
  - name: has_caption
    dtype: bool
  - name: has_transcription
    dtype: bool
  - name: language
    dtype: string
  - name: is_instrumental
    dtype: bool
  - name: nsfw_gore_sim
    dtype: float64
  - name: nsfw_sexual_sim
    dtype: float64
  - name: nsfw_hate_sim
    dtype: float64
  - name: nsfw_gore_label
    dtype: string
  - name: nsfw_sexual_label
    dtype: string
  - name: nsfw_hate_label
    dtype: string
  - name: nsfw_overall_label
    dtype: string
  - name: predicted_play_count
    dtype: float64
  - name: predicted_upvote_count
    dtype: float64
  splits:
  - name: train
    num_examples: 1429734
---

# REDACTED-Tunes

**1,429,734 AI-generated music tracks** from 3 platforms (Suno, Mureka, Udio). Annotated with captions, transcriptions, embeddings, aesthetics scores, and NSFW safety labels. Includes a full-text and vector search engine with a web UI featuring both a beginner-friendly Simple mode and a power-user Advanced mode.

## Quick Stats

| Metric | Value |
|--------|-------|
| Total tracks | **1,429,734** |
| Subsets | Suno (1,037,381), Mureka (383,549), Udio (8,804) |
| Has caption (Music-Whisper) | 1,356,009 (94.8%) |
| Has transcription (Parakeet ASR) | 1,041,488 (72.8%) |
| Instrumental | 388,301 (27.2%) |
| NSFW flagged (very likely + likely) | 23,591 (1.65%) |
| FAISS vector indices | 6 (tag, whisper, caption, transcription, lyric, mood) |
| BM25 text indices | 3 (tags, caption, transcription) |
| Average aesthetics score | 3.29 / 5.0 |
| Total audio duration | ~68,471 hours |

## Dataset Description

REDACTED-Tunes is a curated metadata and annotation dataset covering publicly available AI-generated music from Suno, Udio, and Mureka.

**This dataset does NOT contain audio files.** It contains metadata, annotations, embeddings, and search indices. Audio URLs pointing to the original hosting platforms are included for reference.

### Source Data

The track metadata was collected from three AI music generation platforms:

- **Suno** (1,037,381 tracks): Usernames were discovered via the [nyuuzyou/suno](https://huggingface.co/datasets/nyuuzyou/suno) dataset, which catalogued Suno user profiles found through search queries. Each username corresponds to a public profile on [suno.com](https://suno.com), which lists all of the user's publicly shared songs with direct MP3 download links. Track metadata (title, tags, play count, duration, etc.) was collected from these public profiles.

- **Udio** (8,804 tracks): Audio files were sourced from the [blanchon/udio_dataset](https://huggingface.co/datasets/blanchon/udio_dataset). Track metadata (title, tags, audio URLs) was derived from the dataset's metadata to locate the corresponding public pages on udio.com.

- **Mureka** (383,549 tracks): Metadata was collected from publicly available tracks on the [Mureka](https://mureka.ai) platform. Audio URLs point directly to the platform's public CDN.

### What's Included

For each track:
- **Metadata**: title, tags, genre, mood, duration, play count, upvote count
- **Music-Whisper Caption**: AI-generated music description using [Music Whisper](https://anonymous-hf.up.railway.app/a/3nbwl9ikw488/)
- **Parakeet ASR Transcription**: vocal text for ~1,041,488 tracks
- **Sentence Embeddings**: 768-dim embeddings via [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) for tags, captions, transcriptions, lyrics, and moods
- **Whisper Audio Embeddings**: 768-dim mean-pooled encoder embeddings from Music-Whisper
- **Aesthetics Scores**: coherence, musicality, memorability, clarity, naturalness
- **NSFW Safety Labels**: three-tier classification across gore, sexual, and hate speech dimensions
- **Language Detection**: Detected language of vocal content
- **Pre-built Search Indices**: FAISS vector indices and BM25 text indices ready to serve

### Annotation Pipeline

1. **Music-Whisper** (`REDACTED/music-whisper`): Generates music captions
2. **Parakeet TDT 0.6B** (`nvidia/parakeet-tdt-0.6b-v3`): ASR transcription
3. **EmbeddingGemma 300M** (`google/embeddinggemma-300m`): 768-dim sentence embeddings
4. **Whisper Encoder Embeddings**: Mean-pooled encoder hidden states for audio similarity
5. **Aesthetics MLP**: 5-head MLP predicting coherence / musicality / memorability / clarity / naturalness
6. **NSFW Classification**: Cosine similarity against reference prompts
7. **Language Detection**: `langdetect` on ASR transcriptions

## Web UI

The search engine includes a single-page dark-mode web interface (`index.html`) with both a beginner-friendly **Simple mode** and a power-user **Advanced mode**.

**Simple mode** provides a clean search bar with relevance ranking, language filters, and an optional negative prompt for excluding terms.

**Advanced mode** exposes full control:
- Multiple search modes: Vector Similarity, BM25 Text, Combined, and Music Similarity (audio upload)
- Multiple vector fields: caption, tag, lyric, mood, transcription
- Two-stage refinement: combine different search strategies for precision
- Negative prompts with adjustable weight
- Granular filters: subset, vocal/instrumental, duration, aesthetics, NSFW safety, language

## Data Format

### Parquet Files (`public/`)

Each parquet file corresponds to one TAR file from the source collection. The base-subset parquets (Mureka, Udio, original Suno) contain the full schema including tag/lyric/mood embeddings; the extended Suno parquets (`suno_*_ai_generated_songs*.tar.parquet`) contain annotation-only columns (captions, transcriptions, caption/transcription embeddings, aesthetics, NSFW).

Common columns:

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Filename within the source TAR |
| `tar_file` | str | Source TAR filename |
| `audio_url` | str | Original audio URL (mp3/m4a/ogg) |
| `subset` | str | Source platform (suno/udio/mureka) |
| `title` | str | Track title |
| `tags_text` | str | Comma-separated genre/style tags |
| `mood_text` | str | Mood tags |
| `duration_seconds` | float | Track duration |
| `play_count` | int | Play count on source platform (base subset) |
| `upvote_count` | int | Like/upvote count (base subset) |
| `predicted_play_count` | float | ML-predicted play count (extended Suno) |
| `predicted_upvote_count` | float | ML-predicted upvote count (extended Suno) |
| `music_whisper_caption` | str | Music-Whisper generated caption |
| `parakeet_transcription` | str | Parakeet ASR transcription (plain text) |
| `parakeet_transcription_with_timestamps` | str | ASR with word-level timestamps (base subset) |
| `tag_embedding` | list[float] | 768-dim EmbeddingGemma embedding of tags (base subset) |
| `caption_embedding` | list[float] | 768-dim EmbeddingGemma embedding of caption |
| `transcription_embedding` | list[float] | 768-dim EmbeddingGemma embedding of transcription |
| `lyric_embedding` | list[float] | 768-dim EmbeddingGemma embedding of lyrics (base subset) |
| `mood_embedding` | list[float] | 768-dim EmbeddingGemma embedding of mood (Mureka only) |
| `score_coherence` / `score_musicality` / `score_memorability` / `score_clarity` / `score_naturalness` | float | Aesthetics sub-scores (1 – 5) |
| `score_average` | float | Mean of the five aesthetics sub-scores |
| `has_caption` / `has_transcription` / `is_instrumental` | bool | Annotation flags |
| `language` | str | Detected language of vocal content |
| `nsfw_gore_sim` / `nsfw_sexual_sim` / `nsfw_hate_sim` | float | NSFW cosine similarities |
| `nsfw_gore_label` / `nsfw_sexual_label` / `nsfw_hate_label` / `nsfw_overall_label` | str | NSFW labels |

### Whisper Embeddings (`whisper_embeddings/`)

NPZ files containing mean-pooled Whisper encoder hidden states:
- `embeddings`: float32 array of shape `(N, 768)` – L2-normalized
- `filenames`: string array of filenames matching the parquet entries

### SQLite Database (`search_index/metadata.db`)

The `tracks` table contains all **1,429,734** tracks with 36 columns including metadata, aesthetics scores, predicted engagement (for extended Suno tracks), annotation flags, NSFW safety labels, language codes, and instrumental flags. The `row_id` column is the primary key used by all FAISS indices.

### FAISS Indices (`search_index/faiss_*.index`)

All indices are `IndexFlatIP` (inner product / cosine similarity for L2-normalized vectors) with 768 dimensions. Each index has a corresponding `idmap_*.npy` that maps FAISS internal indices to SQLite `row_id` values.

| Index | Vectors | Coverage | Description |
|-------|---------|----------|-------------|
| `faiss_whisper` | 402,649 | Base subset tracks | Audio encoder embeddings (music similarity) |
| `faiss_caption` | 1,036,860 | Tracks with captions | Music-Whisper caption embeddings |
| `faiss_transcription` | 885,817 | Tracks with transcription | ASR transcription embeddings |
| `faiss_tag` | 402,649 | Base subset tracks | Tag text embeddings |
| `faiss_lyric` | 17,292 | Tracks with lyrics | Lyrics embeddings |
| `faiss_mood` | 383,616 | Mureka only | Mood text embeddings |

Note: `faiss_tag`, `faiss_lyric`, and `faiss_mood` cover only the base subset because the extended Suno source did not provide comparable free-form tag / lyric / mood text fields. Search queries on those fields therefore only hit the base subset; search on `caption`, `transcription`, and `whisper` covers the full dataset.

### BM25 Indices (`search_index/bm25_*.pkl`)

| Index | Documents | Coverage |
|-------|-----------|----------|
| `bm25_caption` | 1,356,009 | Tracks with captions |
| `bm25_transcription` | 1,003,949 | Tracks with transcriptions |
| `bm25_tags` | 401,269 | Base subset tracks |

## Repository Structure

```
REDACTED-tunes/
├── README.md                          # This file
├── server.py                          # FastAPI search server
├── index.html                         # Web UI (dark-mode, single-page app)
├── build_search_index.py              # Index builder script
├── update_indices.py                  # Incremental index updater
├── nsfw_safety_report.html            # Interactive NSFW analysis report
├── nsfw_analysis_data.json            # Raw NSFW analysis data
├── REDACTED-tunes-report.txt             # Dataset statistics report
│
├── public/                            # Annotated metadata parquets
│   ├── mureka_000000.tar.parquet      # One parquet per source TAR file
│   └── ...                            # 370 parquet files total
│
├── search_index/                      # Pre-built search indices
│   ├── metadata.db                    # SQLite database (1,429,734 tracks)
│   ├── faiss_whisper.index            # FAISS IndexFlatIP - audio embeddings
│   ├── faiss_caption.index            # FAISS IndexFlatIP - caption embeddings
│   ├── faiss_transcription.index      # FAISS IndexFlatIP - transcription embeddings
│   ├── faiss_tag.index                # FAISS IndexFlatIP - tag embeddings
│   ├── faiss_lyric.index              # FAISS IndexFlatIP - lyric embeddings
│   ├── faiss_mood.index               # FAISS IndexFlatIP - mood embeddings
│   ├── idmap_*.npy                    # Row ID mappings for each FAISS index
│   ├── bm25_caption.pkl               # BM25 text index for captions
│   ├── bm25_transcription.pkl         # BM25 text index for transcriptions
│   └── bm25_tags.pkl                  # BM25 text index for tags
│
└── whisper_embeddings/                # Raw Whisper encoder embeddings
    ├── mureka_000000.npz              # One NPZ per source TAR file
    └── ...                            # 370 NPZ files total
```

## NSFW Safety Labels

Each track has NSFW safety scores and labels across three dimensions. The classification is performed by computing cosine similarity between the track's transcription embedding and curated reference prompts for each NSFW category.

### Classification Method

1. For each track with a transcription, the 768-dim EmbeddingGemma embedding is compared against reference prompt embeddings for three categories: gore/violence, sexual content, and hate speech
2. Cosine similarity scores are computed for each category
3. Two thresholds per category define the three-tier labeling:
   - **very_likely_nsfw**: cosine similarity above the strict threshold
   - **likely_nsfw**: cosine similarity between strict and moderate thresholds
   - **likely_sfw**: cosine similarity below the moderate threshold
4. The `nsfw_overall_label` is conservative: the worst (most NSFW) label across all three dimensions is used

### Thresholds and Distribution (over all 1,429,734 tracks)

| Dimension | Strict Threshold | Moderate Threshold | Very Likely NSFW | Likely NSFW |
|-----------|-----------------|-------------------|-----------------|-------------|
| Gore/Violence | >= 0.3779 | >= 0.3540 | 5,487 (0.38%) | 5,343 (0.37%) |
| Sexual Content | >= 0.3584 | >= 0.3234 | 4,067 (0.28%) | 4,422 (0.31%) |
| Hate Speech | >= 0.3633 | >= 0.3382 | 4,044 (0.28%) | 4,507 (0.32%) |
| **Overall (conservative)** | – | – | **12,081 (0.85%)** | **11,510 (0.81%)** |

Tracks without a transcription are labeled `likely_sfw` for all dimensions by default (there is no vocal content to flag).

### NSFW Fields in the Dataset

| Field | Type | Description |
|-------|------|-------------|
| `nsfw_gore_sim` | float | Raw cosine similarity for gore/violence |
| `nsfw_sexual_sim` | float | Raw cosine similarity for sexual content |
| `nsfw_hate_sim` | float | Raw cosine similarity for hate speech |
| `nsfw_gore_label` / `nsfw_sexual_label` / `nsfw_hate_label` | str | `very_likely_nsfw` / `likely_nsfw` / `likely_sfw` |
| `nsfw_overall_label` | str | Conservative overall label (worst of the three) |

The raw cosine similarity scores are stored so you can apply your own thresholds. The `nsfw_safety_report.html` file in this repository provides an interactive visual analysis of the NSFW distribution.

### Filtering Behavior in the UI

- **SFW Only** (default in UI): Keeps only tracks with `nsfw_overall_label = likely_sfw` (excludes ~1.65% of tracks)
- **NSFW Only**: Keeps only tracks with `nsfw_overall_label != likely_sfw`
- **All**: No filtering

## Dataset Analyses

These analyses were computed directly from `search_index/metadata.db` over all 1,429,734 tracks. All numbers are reproducible — every query is a single SQL on the released DB; the methodology lives at the end of this section. None of these analyses involve re-running annotation models; they aggregate fields that are already in the dataset.

### A. Genre Distribution (`genre_tags` taxonomy)

Each track is tagged with zero, one, or several entries from a controlled genre taxonomy (substring-matching of `tags_text` and source metadata against canonical genre names). Tags are stored as a JSON array in `genre_tags`. **321,789 tracks (22.5%)** carry at least one genre tag; the remaining ~1.1M (mostly extended Suno without tag-text-rich source metadata) are not auto-classified by genre.

Top 25 genres globally (a track may belong to multiple):

| Genre | Tracks | % of corpus |
|---|---:|---:|
| Latin | 82,146 | 5.7 % |
| R&B / Soul | 69,453 | 4.9 % |
| Electronic / EDM | 67,353 | 4.7 % |
| Hip Hop / Rap | 46,922 | 3.3 % |
| Rock | 31,130 | 2.2 % |
| Metal | 18,334 | 1.3 % |
| Disco / Funk | 17,215 | 1.2 % |
| Punk | 15,051 | 1.1 % |
| Experimental | 13,817 | 1.0 % |
| Country / Americana | 12,382 | 0.9 % |
| Folk / Acoustic | 11,659 | 0.8 % |
| World / Ethnic | 9,144 | 0.6 % |
| Classical | 5,492 | 0.4 % |
| Jazz | 5,137 | 0.4 % |
| Reggae / Dub | 4,190 | 0.3 % |
| Blues | 3,424 | 0.2 % |
| Ambient / Drone | 2,307 | 0.2 % |
| Lo-Fi / Chill | 1,850 | 0.1 % |
| Soundtrack / Score | 1,807 | 0.1 % |
| Pop | 1,673 | 0.1 % |
| House | 997 | 0.1 % |
| A Cappella / Vocal | 498 | < 0.1 % |
| Drum & Bass / Jungle | 469 | < 0.1 % |
| Techno | 444 | < 0.1 % |
| Industrial | 435 | < 0.1 % |

**Top genre per platform** (signals what each generator is most-prompted for):

| Platform | Top 5 genres (counts) |
|---|---|
| Suno | Rock (1.8 K), Hip Hop / Rap (1.7 K), Electronic / EDM (1.5 K), Punk (1.2 K), Metal (1.0 K) |
| Mureka | Latin (81.7 K), R&B / Soul (68.0 K), Electronic / EDM (63.8 K), Hip Hop / Rap (44.1 K), Rock (26.0 K) |
| Udio | Rock (3.3 K), Electronic / EDM (2.1 K), Ambient / Drone (1.2 K), Hip Hop / Rap (1.2 K), Folk / Acoustic (1.1 K) |

The genre distribution is dominated by Mureka, which contributes the vast majority of genre-tagged tracks. Mureka is unusually heavy on Latin (Spanish/Portuguese-language tracks dominate its catalog). Suno and Udio genre counts are lower because most Suno tracks come from the extended collection which lacks tag-text-rich source metadata.

### B. When Were Tracks Generated?

We can only recover generation timestamps for **Mureka**: its `audio_url` paths embed the generation date as `YYYYMMDD` (e.g. `…/audio/20250706/music_…`). All other platforms use opaque UUID-only paths and we have no reliable per-track timestamp. **383,523 of 383,549 Mureka tracks** parse successfully:

| Year-Month | Mureka tracks | % of Mureka |
|---|---:|---:|
| 2024-08 | 12 | 0.0 % |
| 2024-09 | 80 | 0.0 % |
| 2024-10 | 493 | 0.1 % |
| 2024-11 | 10,609 | 2.8 % |
| 2024-12 | 20,350 | 5.3 % |
| 2025-01 | 32,584 | 8.5 % |
| 2025-02 | 45,316 | 11.8 % |
| 2025-03 | 64,488 | 16.8 % |
| 2025-04 | 55,332 | 14.4 % |
| 2025-05 | 61,981 | 16.2 % |
| 2025-06 | 77,654 | 20.2 % |
| 2025-07 | 14,624 | 3.8 % |

Mureka generation date range: **2024-08-10 to 2025-07-06**. The distribution shows accelerating adoption through 2025, with a sharp tail-off in July 2025 because that's when the scraping window closed. For Suno and Udio, we do not have reliable per-track timestamps and deliberately do **not** publish a "year_month" column for those platforms.

### C. Duration Across Platforms

| Platform | N | Mean | Median | P10 | P90 | < 60 s | > 240 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Suno | 1,037,381 | 188.2 s | 185.4 s | 118.6 s | 241.4 s | 14,758 | 112,578 |
| Mureka | 383,549 | 130.7 s | 138.2 s | 48.1 s | 194.7 s | 48,357 | 3,731 |
| Udio | 8,804 | 133.4 s | 131.0 s | 32.8 s | 261.0 s | 2,430 | 1,016 |

Suno's distribution is the widest — a small number of tracks > 4 minutes (112 K) coexists with a tight median around 3 min. Mureka is the shortest on average (median 2:18) and has a long left tail (12.6 % under one minute), reflecting Mureka's preview-style generation defaults. Udio shows broad variance — both short clips and long-form pieces. The web UI's default `min_duration = 60 s` filter cleanly removes the ~65 K-track preview-clip floor without affecting the bulk of any platform.

### D. Language Distribution Per Platform

Detected by `langdetect` on the first 300 characters of the Parakeet ASR transcription. Tracks without intelligible vocals are labeled `unknown`.

**Suno** (n=1,037,381):

| Language | Count | % |
|---|---:|---:|
| en | 580,478 | 56.0 % |
| unknown | 151,721 | 14.6 % |
| es | 51,431 | 5.0 % |
| id | 36,160 | 3.5 % |
| pt | 34,906 | 3.4 % |
| ru | 32,661 | 3.1 % |
| fr | 31,402 | 3.0 % |
| de | 28,605 | 2.8 % |

**Mureka** (n=383,549):

| Language | Count | % |
|---|---:|---:|
| unknown | 235,741 | 61.5 % |
| en | 84,770 | 22.1 % |
| es | 34,778 | 9.1 % |
| pt | 13,082 | 3.4 % |
| fr | 4,129 | 1.1 % |
| id | 2,528 | 0.7 % |
| sw | 1,724 | 0.4 % |
| it | 1,580 | 0.4 % |

Mureka's high `unknown` rate is partly because it generates many short instrumental / preview-style tracks, partly because its catalog is heavily Latin / Spanish-language and our `langdetect` window of 300 chars is sometimes too short to disambiguate.

**Udio** (n=8,804):

| Language | Count | % |
|---|---:|---:|
| en | 6,323 | 71.8 % |
| unknown | 1,271 | 14.4 % |
| es | 239 | 2.7 % |
| de | 217 | 2.5 % |
| fr | 191 | 2.2 % |
| it | 83 | 0.9 % |
| pt | 82 | 0.9 % |
| ru | 80 | 0.9 % |

### E. Aesthetics Scores Per Platform

Five SongEval dimensions on a 1.0 – 5.0 scale, plus the overall `score_average`. Higher is better. All values are means over each platform's full track set.

| Platform | Coherence | Musicality | Memorability | Clarity | Naturalness | **Average** |
|---|---:|---:|---:|---:|---:|---:|
| Suno | 3.45 | 3.26 | 3.35 | 3.17 | 3.15 | **3.28** |
| Mureka | **3.54** | **3.34** | **3.42** | **3.22** | **3.18** | **3.34** |
| Udio | 3.29 | 3.04 | 3.14 | 2.98 | 2.94 | 3.08 |

Mureka leads on every dimension by a small but consistent margin; Udio scores lowest. The five SongEval dimensions are highly correlated within each track (i.e., good tracks tend to be good on all five at once), so the per-dimension ordering across platforms mirrors the overall ordering.

Bucketed `score_average` distribution over all 1,429,734 tracks:

| Bucket | Tracks | % |
|---|---:|---:|
| [1.0, 2.0) | 28,657 | 2.0 % |
| [2.0, 2.5) | 176,170 | 12.3 % |
| [2.5, 3.0) | 314,986 | 22.0 % |
| [3.0, 3.5) | 341,181 | 23.9 % |
| [3.5, 4.0) | 302,933 | 21.2 % |
| [4.0, 4.5) | 230,168 | 16.1 % |
| [4.5, 5.0) | 35,597 | 2.5 % |

Roughly Gaussian centered on ~3.29, with a slightly heavier left tail than right.

### F. Do the Predicted Aesthetics Correlate with Human Reactions?

**Important framing:** we do **not** have SongEval human ratings on the REDACTED-Tunes tracks themselves — SongEval is the *training* dataset for the aesthetics model (2,399 separate songs by the ASLP@NPU group). So we can't directly compare model-vs-rater on these tracks. The closest thing to "human reactions" we have is **real platform play_count and upvote_count** — observable engagement on Suno and Udio (Mureka exposes neither stat).

We sampled tracks across non-Mureka platforms with at least one play, computed `log1p(play_count)` and `log1p(upvote_count)` (raw counts are pathologically long-tailed), and ran Pearson correlations against each aesthetics dimension:

| Aesthetics dimension | r vs log(plays) | r vs log(upvotes) |
|---|---:|---:|
| `score_average` | +0.008 | +0.045 |
| `score_coherence` | +0.007 | +0.044 |
| `score_musicality` | +0.007 | +0.044 |
| `score_memorability` | +0.009 | +0.045 |
| `score_clarity` | +0.008 | +0.044 |
| `score_naturalness` | +0.008 | +0.045 |

**The correlation is essentially zero.** This is itself an interesting finding, and not a bug. Predicted aesthetic quality and platform virality are nearly independent because real-world engagement is dominated by factors the audio doesn't carry: artist brand, platform promotion, time-on-platform, genre niche, prompt-trend cycles, social shares. A separate model trained directly on `(audio -> log_plays)` (REDACTED's `music-popularity` head) achieves log-Pearson ~0.41, telling us that audio *is* informative about engagement — but the **aesthetics signal alone is not the same dimension** as engagement. This separation is a feature, not a flaw: it means using `score_average` as a quality filter doesn't accidentally bias toward popular content, and the two scores can be combined as orthogonal signals during downstream training.

If a researcher wants a true human-evaluation comparison they should run their own listener study on a subsample. The infrastructure is in place: every track has aesthetics scores, NSFW labels, transcription embeddings, and stable `row_id`s suitable for evaluator software.

### G. NSFW by Platform

Fraction of each platform that's flagged at any NSFW level (`very_likely_nsfw + likely_nsfw`):

| Platform | N | Gore | Sexual | Hate | **Overall** |
|---|---:|---:|---:|---:|---:|
| Suno | 1,037,381 | 0.96 % | 0.60 % | 0.64 % | **1.88 %** |
| Mureka | 383,549 | 0.20 % | 0.56 % | 0.47 % | **1.01 %** |
| Udio | 8,804 | 1.10 % | 1.35 % | 1.28 % | **3.13 %** |

Per-platform NSFW shape:
- **Udio** has the highest overall NSFW rate (3.13 %), with all three categories elevated — the platform's user base prompts more for explicit content than the others.
- **Suno** is second (1.88 %) and is unusually skewed toward **gore/violence**: 0.96 % of Suno tracks trip the gore axis, more than 4x the Mureka rate. Likely reflects Suno's popularity for metal / horror / aggressive-rap subgenres.
- **Mureka** is the safest of the three platforms (1.01 %), with sexual-content the dominant dimension when flagged.

Aesthetics x NSFW:

| `nsfw_overall_label` | N | Mean `score_average` |
|---|---:|---:|
| `likely_sfw` | 1,406,101 | 3.297 |
| `likely_nsfw` | 11,510 | 3.083 |
| `very_likely_nsfw` | 12,081 | 3.035 |

NSFW-flagged tracks score modestly *lower* on aesthetics (-0.26 average vs SFW). This is consistent with NSFW content skewing toward genres with rougher production (lo-fi rap, harsh vocals, distorted metal) — but the gap is small enough that you cannot use aesthetics scores as a proxy for safety filtering.

NSFW rate by detected language (>= 5 K tracks per language):

| Language | Total | NSFW | % NSFW |
|---|---:|---:|---:|
| en | 671,571 | 19,996 | 2.98 % |
| tl | 6,361 | 143 | 2.25 % |
| de | 30,283 | 532 | 1.76 % |
| ru | 33,496 | 432 | 1.29 % |
| it | 18,674 | 220 | 1.18 % |
| pt | 48,070 | 377 | 0.78 % |
| es | 86,448 | 653 | 0.76 % |
| fr | 35,722 | 255 | 0.71 % |
| sv | 5,227 | 36 | 0.69 % |
| pl | 13,663 | 91 | 0.67 % |

English is highest at 3.0 %, partly because the NSFW reference prompts used by the classifier are English-only — non-English NSFW content is almost certainly under-flagged. This is a known limitation we surfaced in the **NSFW Safety Labels** section.

### H. Cross-Feature: Vocal/Instrumental and Genre x Aesthetics

**Vocal vs instrumental:**

| | N | Mean `score_average` |
|---|---:|---:|
| Has vocals (`is_instrumental=0`) | 1,041,433 | **3.349** |
| Instrumental (`is_instrumental=1`) | 388,259 | 3.143 |

Vocal tracks score ~0.21 higher on average. This likely reflects two effects: (1) the Music-Whisper encoder was fine-tuned on a Suno-heavy dataset where vocal tracks dominate, so instrumental representation may be slightly weaker, and (2) the SongEval rater protocol rewards "clarity of vocal delivery" and "natural breathing" — dimensions that don't apply to instrumentals and may bias scores downward.

**Top 10 genres by mean `score_average`** (>= 5,000 tracks per genre):

| Genre | Mean | N |
|---|---:|---:|
| Country / Americana | 3.528 | 12,382 |
| Folk / Acoustic | 3.460 | 11,659 |
| Disco / Funk | 3.410 | 17,215 |
| R&B / Soul | 3.404 | 69,453 |
| Hip Hop / Rap | 3.375 | 46,922 |
| Jazz | 3.344 | 5,137 |
| World / Ethnic | 3.343 | 9,144 |
| Rock | 3.324 | 31,130 |
| Punk | 3.306 | 15,051 |
| Electronic / EDM | 3.297 | 67,353 |

**Bottom 5 genres by mean `score_average`** (>= 5,000 tracks per genre):

| Genre | Mean | N |
|---|---:|---:|
| Experimental | 3.160 | 13,817 |
| Classical | 3.213 | 5,492 |
| Latin | 3.222 | 82,146 |
| Metal | 3.255 | 18,334 |
| Electronic / EDM | 3.297 | 67,353 |

The pattern is striking: **vocal-foregrounded, performance-style genres** (Country, Folk, R&B, Blues) score highest; **electronic / abstract / atmospheric** genres (Experimental, Classical, Metal) score lowest. This is again partly a SongEval-protocol artifact (the rater rubric weights *Naturalness* of vocal performance heavily) and partly real: synthetic tracks in vocal-driven genres tend to be more polished, while synthetic ambient and electronic tracks expose synthesis artifacts more.

### Methodology

All numbers above are from a single live SQL pass over `search_index/metadata.db` (SQLite, 36 columns x 1,429,734 rows), using the unmodified released schema:

- Genre tags parsed from `genre_tags` (JSON array string); platform-conditioned via `subset` join.
- Mureka generation dates regex-extracted from `audio_url` (`/audio/(\d{8})/`).
- Durations from `duration_seconds`; percentiles via NumPy on the full per-platform array.
- Language from the `language` column (`langdetect` over first 300 characters of Parakeet transcription).
- Aesthetics correlations: sample of tracks where `play_count > 0` and `subset != 'mureka'`; Pearson `r` of per-dimension score against `log1p(count)`.
- NSFW counts: literal SQL `COUNT(*) WHERE nsfw_*_label IN ('likely_nsfw','very_likely_nsfw')`.
- All cross-feature aggregations restricted to non-NULL `score_average`.

To reproduce, clone the repository, `pip install` the prerequisites, and run the queries against the bundled `metadata.db` — none of these analyses require the FAISS indices or the model weights.

## Running the Search Server

### Prerequisites

```bash
pip install fastapi uvicorn faiss-cpu numpy pandas sentence-transformers torch scipy tqdm python-multipart transformers
```

> **Important — embedder compatibility.** The shipped FAISS caption / transcription / tag / lyric / mood indices were built with the PyTorch [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) model via `sentence-transformers` (which applies the model's specific LastToken pooling and task prompts). You must query the indices with the **same** model in the **same** configuration, otherwise retrieval lands in a different embedding subspace and returns semantically random results. `google/embeddinggemma-300m` is a **gated** model, so accept the terms on its HF page and export an `HF_TOKEN` before running the server.

### Option 1 (default, recommended): PyTorch `google/embeddinggemma-300m`

```bash
# Requires HF_TOKEN for the gated google/embeddinggemma-300m model
HF_TOKEN=your_token python server.py --port 7860 --gpu 0
```

Loads `google/embeddinggemma-300m` via `sentence-transformers` in `bfloat16` on the first available CUDA device (or CPU if CUDA is absent). This is the configuration that matches the shipped FAISS indices. On GPU, query embedding takes ~15 ms; on CPU, ~430 ms.

### Option 2 (same model, fastest on CPU): HF Text Embeddings Inference

TEI runs `google/embeddinggemma-300m` in an optimized server container and provides ~25 ms CPU embeddings. Same model, same pooling — fully compatible with the indices.

```bash
docker run -d --name tei-embeddings \
  -p 8090:80 \
  -e HF_TOKEN=your_token \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id google/embeddinggemma-300m \
  --max-batch-requests 4

python server.py --port 7860 --gpu 0 --tei-url http://localhost:8090
```

### Option 3 (opt-in, NOT compatible with shipped indices): ONNX quantized

The non-gated [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) repo does **not** ship a `sentence-transformers` config, so `SentenceTransformer` falls back to naive mean-pooling — yielding a different subspace than the one the FAISS indices were built in. Using ONNX against the shipped indices produces ~0.04 cosine on a query that should score ~0.9, and nonsense search results.

Only enable this path if you **rebuild** the FAISS indices locally with the same ONNX embedder + mean pooling:

```bash
REDACTEDTUNES_USE_ONNX_EMBEDDER=1 python server.py --port 7860 --no-whisper
# The server will print a WARNING at startup reminding you retrieval will be degraded on the shipped indices.
```

### Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 7860 | HTTP port |
| `--host` | 0.0.0.0 | Bind address |
| `--gpu` | 0 | GPU ID for the text embedder and the Whisper encoder |
| `--tei-url` | None | TEI server URL for text embeddings (skips loading the local embedder) |
| `--no-whisper` | False | Skip loading the Music-Whisper encoder (disables audio-upload similarity search) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | – | Required for `google/embeddinggemma-300m` (and for `REDACTED/music-whisper` if audio-upload search is enabled). |
| `REDACTEDTUNES_USE_ONNX_EMBEDDER` | `0` | Set to `1` to opt into the ONNX embedder. Retrieval quality on the shipped FAISS indices will be degraded — only set if you have rebuilt the indices with the same ONNX embedder. |
| `REDACTEDTUNES_SKIP_FAISS` | – | Comma-separated list of FAISS fields to *not* load at startup (e.g. `tag,lyric,mood`). Useful for memory-constrained deployments. Skipped fields still accept query requests but return 404-style empty results. |
| `REDACTEDTUNES_SKIP_BM25` | – | Comma-separated list of BM25 fields to skip (e.g. `tags,transcription`). |

### What Loads at Startup

1. **6 FAISS indices** (tag, whisper, caption, transcription, lyric, mood) — unless listed in `REDACTEDTUNES_SKIP_FAISS`
2. **3 BM25 indices** (tags, caption, transcription) — unless listed in `REDACTEDTUNES_SKIP_BM25`
3. **SQLite database** (1,429,734 tracks)
4. **Text embedder** — in order of preference: TEI if `--tei-url` is set; otherwise `google/embeddinggemma-300m` via SentenceTransformer (default); otherwise `onnx-community/embeddinggemma-300m-ONNX` only if `REDACTEDTUNES_USE_ONNX_EMBEDDER=1`.
5. **Music-Whisper encoder** (optional, on GPU): for audio-upload similarity search

Total memory: ~25 GB RAM + ~1 GB GPU VRAM when Whisper + EmbeddingGemma are both loaded on GPU.

## Search API Reference

The FastAPI server exposes the following endpoints.

### `GET /`
Serves the HTML search frontend.

### `GET /nsfw-report`
Serves the interactive NSFW safety analysis report.

---

### `POST /api/search`

Main search endpoint supporting vector similarity, BM25 text search, and combined mode with optional two-stage refinement.

#### Request Body (JSON)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | str | *required* | Search query text |
| `negative_query` | str \| null | null | Negative prompt (subtracted from query embedding in vector mode) |
| `search_type` | str | `"bm25"` | `"vector"` \| `"bm25"` \| `"combined"` |
| `vector_field` | str | `"caption"` | FAISS index: `"tag"` \| `"caption"` \| `"lyric"` \| `"mood"` \| `"transcription"` |
| `bm25_field` | str | `"caption"` | BM25 index: `"tags"` \| `"caption"` \| `"transcription"` |
| `rank_by` | str | `"similarity"` | `"similarity"` \| `"aesthetics"` \| `"plays"` \| `"likes"` |
| `min_aesthetics` | float \| null | null | Minimum aesthetics score (0 – 5 scale) |
| `min_similarity` | float \| null | null | Minimum cosine similarity score |
| `subset_filter` | str \| null | null | `"suno"` \| `"udio"` \| `"mureka"` |
| `vocal_filter` | str \| null | null | `"instrumental"` \| `"vocals"` |
| `min_duration` | float \| null | 60.0 | Minimum duration in seconds |
| `languages` | list[str] \| null | null | Language codes to include (e.g. `["en", "es"]`), null = all |
| `negative_weight` | float | 0.7 | Weight for negative query subtraction (0.0 – 1.0) |
| `nsfw_filter` | str \| null | null | `"sfw_only"` \| `"nsfw_only"` \| null (all) |
| `top_k` | int | 50 | Number of results to return |
| `stage2_enabled` | bool | false | Enable two-stage refinement |
| `stage2_query` | str \| null | null | Query text for Stage 2 |
| `stage2_search_type` | str | `"vector"` | `"vector"` \| `"bm25"` |
| `stage2_vector_field` | str | `"caption"` | Vector field for Stage 2 |
| `stage2_bm25_field` | str | `"caption"` | BM25 field for Stage 2 |
| `stage2_top_k` | int | 50 | Number of results after Stage 2 re-ranking |

#### Example Request

```bash
curl -X POST http://localhost:7860/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "dreamy ambient synth pad",
    "search_type": "vector",
    "vector_field": "caption",
    "rank_by": "similarity",
    "nsfw_filter": "sfw_only",
    "top_k": 20
  }'
```

### `POST /api/search_similar`

Find tracks similar to an existing track by `row_id`, using the Whisper audio embeddings.

### `POST /api/search_by_audio`

Upload an audio file (`multipart/form-data`, first 30 s used) to find similar tracks by audio fingerprint using the Music-Whisper encoder.

### `GET /api/stats`

Returns dataset statistics and index information.

#### Response Body (example)

```json
{
  "total_tracks": 1429734,
  "subsets": {
    "suno": 1037381,
    "mureka": 383549,
    "udio": 8804
  },
  "score_average": { "mean": 3.29, "min": 1.4, "max": 4.77 },
  "with_caption": 1356009,
  "with_transcription": 1041488,
  "faiss_indices": {
    "whisper": 402649,
    "caption": 1036860,
    "transcription": 885817,
    "tag": 402649,
    "lyric": 17292,
    "mood": 383616
  },
  "bm25_indices": {
    "caption": 1356009,
    "transcription": 1003949,
    "tags": 401269
  },
  "instrumental_count": 388301,
  "whisper_embeddings": 402649
}
```

## Result Track Object

Every search endpoint returns results as a list of track objects with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `row_id` | int | Unique track identifier (primary key in SQLite) |
| `title` | str | Track title |
| `audio_url` | str | URL to the audio file on the source platform |
| `subset` | str | Source platform: suno, udio, mureka |
| `tags_text` | str | Comma-separated genre/style tags |
| `mood_text` | str | Mood descriptors |
| `genre_tags` / `scene_tags` / `emotion_tags` | list[str] | Parsed taxonomy tags |
| `score_average`, `score_coherence`, `score_musicality`, `score_memorability`, `score_clarity`, `score_naturalness` | float \| null | Aesthetics scores (1 – 5) |
| `play_count` / `upvote_count` | int | Source-platform stats (0 for extended Suno tracks; use predicted_* instead) |
| `duration_seconds` | float \| null | Track duration in seconds |
| `music_whisper_caption` | str | AI-generated music description |
| `has_caption` / `has_transcription` / `is_instrumental` | bool | Annotation flags |
| `language` | str | Detected language code (e.g. `"en"`) or `"unknown"` |
| `score` | float \| null | Search relevance score |
| `score_type` | str | `cosine_similarity`, `bm25`, `aesthetics`, `play_count`, `upvote_count` |
| `has_whisper_emb` | bool | Whether the track has a Whisper audio embedding |
| `nsfw_overall_label`, `nsfw_gore_label`, `nsfw_sexual_label`, `nsfw_hate_label` | str | NSFW labels |
| `nsfw_gore_sim`, `nsfw_sexual_sim`, `nsfw_hate_sim` | float \| null | NSFW cosine similarity scores |
| `stage1_score`, `stage2_score` | float | Only present if Stage 2 was enabled |

## Building the Index from Scratch

```bash
python build_search_index.py --force
```

This reads all parquets from `public/`, builds the SQLite database, FAISS indices, and BM25 indices.

## Quick Start

### 1. Clone and download

```bash
git clone https://anonymous-hf.up.railway.app/a/pib3syfuxp02/
cd REDACTED-tunes
```

### 2. Start the search server

```bash
pip install fastapi uvicorn faiss-cpu numpy pandas sentence-transformers torch scipy tqdm python-multipart transformers
HF_TOKEN=your_token python server.py
```

### 3. Open the web UI

Navigate to `http://localhost:7860` in your browser.

## Related Datasets

- **[REDACTED-Tunes RPG Music](https://anonymous-hf.up.railway.app/a/blpsgfq6xdyk/)** — 2,580 instrumental tracks from REDACTED-Tunes (Suno + Udio) annotated with Gemini 3 Flash Preview across 18 RPG genres (high fantasy, cosmic horror, cyberpunk, ...) and evoked-emotion tags. Each genre has its own FAISS index over per-track "situation" lists, so you can search by natural-language scenario (e.g. *"sneaking through a dark dungeon"*). Includes a dedicated FastAPI server (`rpg_server.py`) and purple-themed web UI (`rpg_index.html`).
- **[Music Whisper](https://anonymous-hf.up.railway.app/a/3nbwl9ikw488/)** — Music captioning model
- **[blanchon/udio_dataset](https://huggingface.co/datasets/blanchon/udio_dataset)** — Udio audio files
- **[nyuuzyou/suno](https://huggingface.co/datasets/nyuuzyou/suno)** — Suno user profile metadata

## Models Used

| Model | Purpose | Output |
|-------|---------|--------|
| [Music Whisper](https://anonymous-hf.up.railway.app/a/3nbwl9ikw488/) | Music captioning + audio embeddings | Text caption + 768-dim encoder embedding |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ASR transcription | Text + word-level timestamps |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | Text sentence embeddings | 768-dim L2-normalized vectors |
| [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) | Text embeddings (ONNX, non-gated) | 768-dim L2-normalized vectors (int8 quantized) |

## License

Apache 2.0

## Citation

If you use this dataset, please cite:

```bibtex
@misc{REDACTED-tunes-2026,
  title={REDACTED-Tunes: Annotated AI-Generated Music Metadata Dataset},
  author={REDACTED},
  year={2026},
  url={https://anonymous-hf.up.railway.app/a/pib3syfuxp02/},
}
```
