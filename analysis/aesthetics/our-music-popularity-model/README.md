# Our Music-Popularity Model

Local runner for `laion/music-popularity`, an audio-only predictor of
platform popularity signals from `laion/music-whisper` segment-pooled features.

The model predicts:

- `log1p_play_count`
- `log1p_upvote_count`
- `estimated_play_count`
- `estimated_upvote_count`

The log outputs are the actual model targets; the estimated counts are
`expm1(log1p_count)` convenience fields.

## Build Inputs

This reuses the Audiobox staging set so all audio-only front-ends score the
same files.

```bash
cd analysis
uv run python aesthetics/our-music-popularity-model/build_input.py
```

## Run Scoring

```bash
cd analysis
uv run --with-requirements aesthetics/our-music-popularity-model/requirements.txt \
  python aesthetics/our-music-popularity-model/eval.py \
  -i aesthetics/our-music-popularity-model/input.jsonl \
  -o aesthetics/our-music-popularity-model/output \
  --use_cpu True
```

Outputs:

- `output/result.json`
- `output/metadata.json`

After scoring, run `python main.py` from `analysis/` to compare the popularity
signals against human ratings, Audiobox, SongEval, and our aesthetics model.
