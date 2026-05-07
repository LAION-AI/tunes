# Our Music-Aesthetics Model

Local runner for `laion/music-aesthetics`, a SongEval-calibrated audio-only
aesthetics scorer on top of `laion/music-whisper`.

The model predicts five 1--5 scores:

- `Naturalness`
- `Clarity`
- `Musicality`
- `Coherence`
- `Memorability`

`Overall_Aesthetics` is the arithmetic mean of those five scores.

## Build Inputs

This reuses the Audiobox staging set so all audio-only front-ends score the
same files.

```bash
cd analysis
uv run python aesthetics/our-music-aesthetics-model/build_input.py
```

## Run Scoring

```bash
cd analysis
uv run --with-requirements aesthetics/our-music-aesthetics-model/requirements.txt \
  python aesthetics/our-music-aesthetics-model/eval.py \
  -i aesthetics/our-music-aesthetics-model/input.jsonl \
  -o aesthetics/our-music-aesthetics-model/output \
  --use_cpu True
```

The first run downloads:

- `laion/music-whisper`
- `laion/music-aesthetics/stage1_bottleneck.pt`
- `laion/music-aesthetics/expert_*.pt`

Outputs:

- `output/result.json`: raw 1--5 model scores keyed by `song_id`
- `output/metadata.json`: source/snippet/path metadata for scored rows

The main analysis rescales these scores from 1--5 to 1--10 when comparing
against human ratings and Audiobox.
